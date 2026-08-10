from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


PAPER_VARIANT_CODES = (
    "2223",
    "4223",
    "6223",
    "2323",
    "2313",
    "2413",
    "2412",
    "2411",
)

# The paper does not publish layer-by-layer channel dimensions. These schedules
# follow the canonical ShuffleNetV2 stage widths and are reconstruction choices.
WIDTH_CHANNELS = {
    0.5: (24, 48, 96, 192, 1024),
    1.0: (24, 116, 232, 464, 1024),
    2.0: (24, 244, 488, 976, 2048),
}


@dataclass(frozen=True)
class PalmNetSpec:
    width_mult: float
    variant_code: str
    shuffle_blocks: int
    mobilenetv3_blocks: int
    mbconv_blocks: int
    expansion_factor: int
    channels: tuple[int, int, int, int, int]
    se_ratio: float = 0.25
    drop_path_rate: float = 0.0
    reconstruction_status: str = "paper-constrained independent reconstruction"

    def to_metadata(self) -> dict:
        metadata = asdict(self)
        metadata["channels"] = list(self.channels)
        return metadata


def parse_variant_code(variant_code: str) -> tuple[int, int, int, int]:
    code = str(variant_code)
    if code not in PAPER_VARIANT_CODES:
        raise ValueError(
            f"Unsupported PalmNet variant {code!r}; paper variants are {PAPER_VARIANT_CODES}"
        )
    values = tuple(int(value) for value in code)
    if any(value <= 0 for value in values):
        raise ValueError(f"PalmNet variant digits must be positive: {code!r}")
    return values


def make_palmnet_spec(
    width_mult: float,
    variant_code: str,
    *,
    se_ratio: float = 0.25,
    drop_path_rate: float = 0.0,
) -> PalmNetSpec:
    width = float(width_mult)
    if width not in WIDTH_CHANNELS:
        raise ValueError(f"Unsupported PalmNet width {width}; choose one of {tuple(WIDTH_CHANNELS)}")
    shuffle, mobilenetv3, mbconv, expansion = parse_variant_code(variant_code)
    if not 0.0 <= drop_path_rate < 1.0:
        raise ValueError("drop_path_rate must be in [0, 1)")
    if not 0.0 < se_ratio <= 1.0:
        raise ValueError("se_ratio must be in (0, 1]")
    return PalmNetSpec(
        width_mult=width,
        variant_code=str(variant_code),
        shuffle_blocks=shuffle,
        mobilenetv3_blocks=mobilenetv3,
        mbconv_blocks=mbconv,
        expansion_factor=expansion,
        channels=WIDTH_CHANNELS[width],
        se_ratio=se_ratio,
        drop_path_rate=drop_path_rate,
    )


def _make_divisible(value: float, divisor: int = 8) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return inputs
        keep = 1.0 - self.probability
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = inputs.new_empty(shape).bernoulli_(keep)
        return inputs * mask / keep


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        groups: int = 1,
        activation: type[nn.Module] | None = nn.ReLU,
    ):
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation is not None:
            layers.append(activation(inplace=True))
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, ratio: float):
        super().__init__()
        reduced = max(1, _make_divisible(channels * ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, reduced, 1)
        self.activation = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(reduced, channels, 1)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self.pool(inputs)
        scale = self.reduce(scale)
        scale = self.activation(scale)
        scale = self.expand(scale)
        return inputs * self.gate(scale)


def channel_shuffle(inputs: torch.Tensor, groups: int = 2) -> torch.Tensor:
    batch, channels, height, width = inputs.shape
    inputs = inputs.reshape(batch, groups, channels // groups, height, width)
    inputs = inputs.transpose(1, 2).contiguous()
    return inputs.reshape(batch, channels, height, width)


class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("ShuffleNetV2 stride must be 1 or 2")
        if out_channels % 2 != 0:
            raise ValueError("ShuffleNetV2 output channels must be even")
        if stride == 1 and in_channels != out_channels:
            raise ValueError("Stride-1 ShuffleNetV2 block must preserve channels")

        branch_channels = out_channels // 2
        self.stride = stride
        if stride == 2:
            self.branch1 = nn.Sequential(
                ConvBNAct(
                    in_channels,
                    in_channels,
                    3,
                    stride=2,
                    groups=in_channels,
                    activation=None,
                ),
                ConvBNAct(in_channels, branch_channels, 1),
            )
            branch2_input = in_channels
        else:
            self.branch1 = nn.Identity()
            branch2_input = branch_channels
        self.branch2 = nn.Sequential(
            ConvBNAct(branch2_input, branch_channels, 1),
            ConvBNAct(
                branch_channels,
                branch_channels,
                3,
                stride=stride,
                groups=branch_channels,
                activation=None,
            ),
            ConvBNAct(branch_channels, branch_channels, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            first, second = inputs.chunk(2, dim=1)
            output = torch.cat((first, self.branch2(second)), dim=1)
        else:
            output = torch.cat((self.branch1(inputs), self.branch2(inputs)), dim=1)
        return channel_shuffle(output)


class MobileNetV3Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int,
        stride: int,
        se_ratio: float,
        drop_path_rate: float,
    ):
        super().__init__()
        hidden = _make_divisible(in_channels * expansion)
        self.expand = ConvBNAct(in_channels, hidden, 1)
        self.depthwise = ConvBNAct(hidden, hidden, 3, stride=stride, groups=hidden)
        self.project = ConvBNAct(hidden, out_channels, 1, activation=None)
        # The paper diagram places SE after projection in its MobileNetV3 block.
        self.se = SqueezeExcite(out_channels, se_ratio)
        self.drop_path = DropPath(drop_path_rate)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.expand(inputs)
        output = self.depthwise(output)
        output = self.project(output)
        output = self.se(output)
        if self.use_residual:
            output = inputs + self.drop_path(output)
        return output


class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int,
        stride: int,
        se_ratio: float,
        drop_path_rate: float,
    ):
        super().__init__()
        hidden = _make_divisible(in_channels * expansion)
        self.expand = ConvBNAct(in_channels, hidden, 1, activation=nn.SiLU)
        self.depthwise = ConvBNAct(
            hidden,
            hidden,
            3,
            stride=stride,
            groups=hidden,
            activation=nn.SiLU,
        )
        self.se = SqueezeExcite(hidden, se_ratio)
        self.project = ConvBNAct(hidden, out_channels, 1, activation=None)
        self.drop_path = DropPath(drop_path_rate)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.expand(inputs)
        output = self.depthwise(output)
        output = self.se(output)
        output = self.project(output)
        if self.use_residual:
            output = inputs + self.drop_path(output)
        return output


def _make_stage(block_type, count: int, in_channels: int, out_channels: int, spec: PalmNetSpec):
    blocks: list[nn.Module] = []
    for index in range(count):
        stride = 2 if index == 0 else 1
        block_input = in_channels if index == 0 else out_channels
        if block_type is ShuffleNetV2Block:
            block = block_type(block_input, out_channels, stride)
        else:
            block = block_type(
                block_input,
                out_channels,
                spec.expansion_factor,
                stride,
                spec.se_ratio,
                spec.drop_path_rate,
            )
        blocks.append(block)
    return nn.Sequential(*blocks)


class PalmNet(nn.Module):
    def __init__(self, spec: PalmNetSpec, num_classes: int, input_channels: int):
        super().__init__()
        self.spec = spec
        stem, shuffle_channels, mobile_channels, mbconv_channels, head_channels = spec.channels
        self.stem = nn.Sequential(
            ConvBNAct(input_channels, stem, 3, stride=2),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.shuffle_stage = _make_stage(
            ShuffleNetV2Block, spec.shuffle_blocks, stem, shuffle_channels, spec
        )
        self.mobilenetv3_stage = _make_stage(
            MobileNetV3Block,
            spec.mobilenetv3_blocks,
            shuffle_channels,
            mobile_channels,
            spec,
        )
        self.mbconv_stage = _make_stage(
            MBConvBlock, spec.mbconv_blocks, mobile_channels, mbconv_channels, spec
        )
        self.head = ConvBNAct(mbconv_channels, head_channels, 1, activation=nn.SiLU)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(head_channels, num_classes)

    def reconstruction_metadata(self) -> dict:
        return self.spec.to_metadata()

    def forward_stages(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        stem = self.stem(inputs)
        shuffle = self.shuffle_stage(stem)
        mobilenetv3 = self.mobilenetv3_stage(shuffle)
        mbconv = self.mbconv_stage(mobilenetv3)
        head = self.head(mbconv)
        return stem, shuffle, mobilenetv3, mbconv, head

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_stages(inputs)[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        features = self.pool(features).flatten(1)
        return self.classifier(features)


def build_palmnet(
    *,
    width_mult: float,
    variant_code: str,
    num_classes: int = 834,
    input_channels: int = 3,
    drop_path_rate: float = 0.0,
    se_ratio: float = 0.25,
) -> PalmNet:
    spec = make_palmnet_spec(
        width_mult,
        variant_code,
        se_ratio=se_ratio,
        drop_path_rate=drop_path_rate,
    )
    return PalmNet(spec, num_classes=num_classes, input_channels=input_channels)

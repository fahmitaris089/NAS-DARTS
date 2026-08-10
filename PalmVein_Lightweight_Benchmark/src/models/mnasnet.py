from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MnasNetStageSpec:
    block: str
    expansion: int
    channels: int
    repeats: int
    stride: int
    kernel: int
    se_ratio: float = 0.0


# Figure 7 in the MnasNet paper. The same stage definition is present in the
# authors' TensorFlow TPU implementation referenced by the paper.
MNASNET_A1_STAGES = (
    MnasNetStageSpec("ds", 1, 16, 1, 1, 3),
    MnasNetStageSpec("ir", 6, 24, 2, 2, 3),
    MnasNetStageSpec("ir", 3, 40, 3, 2, 5, 0.25),
    MnasNetStageSpec("ir", 6, 80, 4, 2, 3),
    MnasNetStageSpec("ir", 6, 112, 2, 1, 3, 0.25),
    MnasNetStageSpec("ir", 6, 160, 3, 2, 5, 0.25),
    MnasNetStageSpec("ir", 6, 320, 1, 1, 3),
)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int = 1, groups: int = 1, *, activate: bool = True):
        padding = kernel // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, groups=groups, bias=False),
            # The authors use TensorFlow momentum=0.99. PyTorch defines the
            # update coefficient in the opposite direction, hence 0.01 here.
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        ]
        if activate:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, expanded_channels: int, input_channels: int, ratio: float):
        super().__init__()
        reduced_channels = max(1, int(input_channels * ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(expanded_channels, reduced_channels, 1)
        self.act = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(reduced_channels, expanded_channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.expand(self.act(self.reduce(self.pool(x))))
        return x * self.gate(scale)


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()
        self.depthwise = ConvBNAct(in_channels, in_channels, kernel, stride, groups=in_channels)
        self.pointwise = ConvBNAct(in_channels, out_channels, 1, activate=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class A1InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        expansion: int,
        se_ratio: float,
    ):
        super().__init__()
        expanded_channels = in_channels * expansion
        self.expand = ConvBNAct(in_channels, expanded_channels, 1)
        self.depthwise = ConvBNAct(
            expanded_channels, expanded_channels, kernel, stride, groups=expanded_channels
        )
        self.se = (
            SqueezeExcite(expanded_channels, in_channels, se_ratio)
            if se_ratio > 0
            else nn.Identity()
        )
        self.project = ConvBNAct(expanded_channels, out_channels, 1, activate=False)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.project(self.se(self.depthwise(self.expand(x))))
        return x + value if self.use_residual else value


class MnasNetA1(nn.Module):
    """MnasNet-A1 reconstructed from the paper's published stage definition."""

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.stem = ConvBNAct(3, 32, 3, stride=2)
        stages: list[nn.Module] = []
        in_channels = 32
        for spec in MNASNET_A1_STAGES:
            blocks: list[nn.Module] = []
            for index in range(spec.repeats):
                stride = spec.stride if index == 0 else 1
                if spec.block == "ds":
                    block = DepthwiseSeparable(in_channels, spec.channels, spec.kernel, stride)
                else:
                    block = A1InvertedResidual(
                        in_channels,
                        spec.channels,
                        spec.kernel,
                        stride,
                        spec.expansion,
                        spec.se_ratio,
                    )
                blocks.append(block)
                in_channels = spec.channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.head = ConvBNAct(in_channels, 1280, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.zeros_(module.bias)

    def forward_stages(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = [self.stem(x)]
        for stage in self.stages:
            outputs.append(stage(outputs[-1]))
        return outputs

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = self.forward_stages(x)[-1]
        return self.pool(self.head(value)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


# The implementation below intentionally preserves the former module layout
# and state-dict keys. Existing checkpoints were trained with this topology,
# which is equivalent to torchvision.mnasnet1_0 and is now named explicitly.
MNASNET_B1_TORCHVISION_STAGES = (
    # expansion, channels, repeats, stride, kernel
    (1, 16, 1, 1, 3),
    (3, 24, 3, 2, 3),
    (3, 40, 3, 2, 5),
    (6, 80, 3, 2, 5),
    (6, 96, 2, 1, 3),
    (6, 192, 4, 2, 5),
    (6, 320, 1, 1, 3),
)


class B1InvertedResidual(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int, stride: int, expansion: int):
        super().__init__()
        hidden = in_c * expansion
        layers: list[nn.Module] = []
        if expansion != 1:
            layers += [
                nn.Conv2d(in_c, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, kernel, stride, kernel // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        self.layers = nn.Sequential(*layers)
        self.residual = stride == 1 and in_c == out_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.layers(x)
        return x + value if self.residual else value


class MnasNetB1Torchvision(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        in_c = 32
        for expansion, out_c, repeats, stride, kernel in MNASNET_B1_TORCHVISION_STAGES:
            for index in range(repeats):
                layers.append(
                    B1InvertedResidual(in_c, out_c, kernel, stride if index == 0 else 1, expansion)
                )
                in_c = out_c
        layers += [
            nn.Conv2d(in_c, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.layers(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def _load_torchvision_weights_by_shape(model: MnasNetB1Torchvision) -> None:
    from torchvision.models import MNASNet1_0_Weights, mnasnet1_0

    reference = mnasnet1_0(weights=MNASNet1_0_Weights.IMAGENET1K_V1)
    source = list(reference.state_dict().items())
    target = list(model.state_dict().items())
    source_shapes = [tuple(value.shape) for _, value in source]
    target_shapes = [tuple(value.shape) for _, value in target]
    if source_shapes != target_shapes:
        raise RuntimeError(
            "torchvision.mnasnet1_0 failed the explicit MnasNet-B1 tensor-shape equivalence audit; "
            "pretrained weights were not loaded."
        )
    model.load_state_dict(
        {
            target_name: source_value
            for (target_name, _), (_, source_value) in zip(target, source)
        },
        strict=True,
    )


def build_mnasnet_a1(num_classes: int, pretrained: bool = False) -> MnasNetA1:
    if pretrained:
        raise ValueError(
            "mnasnet_a1 has no audited official PyTorch pretrained weights; "
            "use the controlled scratch protocol"
        )
    return MnasNetA1(num_classes=num_classes)


def build_mnasnet_b1_torchvision(
    num_classes: int, pretrained: bool = False
) -> MnasNetB1Torchvision:
    model = MnasNetB1Torchvision(num_classes=1000 if pretrained else num_classes)
    if pretrained:
        _load_torchvision_weights_by_shape(model)
        model.classifier[1] = nn.Linear(1280, num_classes)
    return model

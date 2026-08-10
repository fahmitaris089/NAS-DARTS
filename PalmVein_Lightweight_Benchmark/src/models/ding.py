from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DingBlockSpec:
    kind: str
    in_channels: int
    bottleneck_channels: int | None
    out_channels: int
    pool: bool


class DingConvBlock(nn.Module):
    """One Conv-BN-ReLU block from the paper's baseline network."""

    def __init__(self, in_channels: int, out_channels: int, *, pool: bool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class DingPointwiseBlock(nn.Module):
    """Paper PW block: 1x1 reduction, 3x3 extraction, 1x1 expansion."""

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        out_channels: int,
        *,
        pool: bool,
    ):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.spatial = nn.Conv2d(
            bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False
        )
        self.expand = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.spatial(x)
        x = self.expand(x)
        return self.pool(self.relu(self.bn(x)))


class DingReconstruction(nn.Module):
    """Paper-constrained independent reconstruction, not authors' source code."""

    def __init__(
        self,
        specs: tuple[DingBlockSpec, ...],
        num_classes: int,
        input_channels: int,
    ):
        super().__init__()
        if len(specs) != 6:
            raise ValueError(f"Ding reconstruction requires six blocks; got {len(specs)}")
        if specs[0].in_channels != 1:
            raise ValueError("Paper specifications must start from one grayscale channel")

        adapted_specs = (
            DingBlockSpec(
                specs[0].kind,
                input_channels,
                specs[0].bottleneck_channels,
                specs[0].out_channels,
                specs[0].pool,
            ),
            *specs[1:],
        )
        blocks: list[nn.Module] = []
        for spec in adapted_specs:
            if spec.kind == "conv3x3":
                blocks.append(
                    DingConvBlock(spec.in_channels, spec.out_channels, pool=spec.pool)
                )
            elif spec.kind == "pw_bottleneck":
                if spec.bottleneck_channels is None:
                    raise ValueError("PW block requires bottleneck_channels")
                blocks.append(
                    DingPointwiseBlock(
                        spec.in_channels,
                        spec.bottleneck_channels,
                        spec.out_channels,
                        pool=spec.pool,
                    )
                )
            else:
                raise ValueError(f"Unsupported Ding block kind: {spec.kind}")

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(adapted_specs[-1].out_channels, num_classes)
        self.architecture_spec = adapted_specs
        self.reconstruction_status = "paper-constrained independent reconstruction"
        self.paper_reference_input_channels = 1
        self.paper_reference_num_classes = 500
        self.benchmark_input_channels = input_channels
        self.num_classes = num_classes

    def forward_block_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return tuple(outputs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.pool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


DING_BASELINE_SPECS = (
    DingBlockSpec("conv3x3", 1, None, 32, True),
    DingBlockSpec("conv3x3", 32, None, 32, True),
    DingBlockSpec("conv3x3", 32, None, 64, True),
    DingBlockSpec("conv3x3", 64, None, 64, True),
    DingBlockSpec("conv3x3", 64, None, 128, True),
    DingBlockSpec("conv3x3", 128, None, 128, False),
)

DING_PW_SPECS = (
    *DING_BASELINE_SPECS[:3],
    DingBlockSpec("pw_bottleneck", 64, 32, 64, True),
    DingBlockSpec("pw_bottleneck", 64, 16, 128, True),
    DingBlockSpec("pw_bottleneck", 128, 64, 128, False),
)

DING_PRUNED_SPECS = (
    DingBlockSpec("conv3x3", 1, None, 22, True),
    DingBlockSpec("conv3x3", 22, None, 22, True),
    DingBlockSpec("conv3x3", 22, None, 44, True),
    DingBlockSpec("pw_bottleneck", 44, 22, 44, True),
    DingBlockSpec("pw_bottleneck", 44, 11, 89, True),
    DingBlockSpec("pw_bottleneck", 89, 44, 89, False),
)


def build_ding_baseline(num_classes: int, input_channels: int = 3):
    return DingReconstruction(DING_BASELINE_SPECS, num_classes, input_channels)


def build_ding_pw(num_classes: int, input_channels: int = 3):
    return DingReconstruction(DING_PW_SPECS, num_classes, input_channels)


def build_ding_pruned(num_classes: int, input_channels: int = 3):
    return DingReconstruction(DING_PRUNED_SPECS, num_classes, input_channels)

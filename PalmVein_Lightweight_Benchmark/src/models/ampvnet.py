from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AMPVNetSpec:
    input_channels: int = 3
    stem_channels: int = 32
    stage_channels: tuple[int, ...] = (64, 128, 256, 512)
    expansion_ratio: int = 4
    dropout: float = 0.2


class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, groups: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      kernel_size // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class AMPVNetDownsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = ConvBNReLU6(channels, channels, 1)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.project(x))


class AMPVNetBottleneck(nn.Module):
    """Figure 9 bottleneck without a shortcut, followed by average pooling."""

    def __init__(self, in_channels: int, out_channels: int,
                 expansion_ratio: int = 4) -> None:
        super().__init__()
        # Figure 9 omits the numeric expansion ratio. A ratio of four gives
        # 1.638M parameters for the paper's 1,100-class setting, within 1.7%
        # of the reported rounded 1.61M total.
        hidden_channels = in_channels * expansion_ratio
        self.expand = ConvBNReLU6(in_channels, hidden_channels, 1)
        self.depthwise = ConvBNReLU6(
            hidden_channels, hidden_channels, 3, groups=hidden_channels
        )
        self.project = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project_bn(self.project(x))
        return self.pool(x)


class AMPVNet(nn.Module):
    """Paper-constrained reconstruction of AMPVNet from Figure 9."""

    reconstruction_status = "paper-constrained independent reconstruction"

    def __init__(self, num_classes: int = 1100, input_channels: int = 3,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.spec = AMPVNetSpec(input_channels=input_channels, dropout=dropout)
        self.stem_conv = ConvBNReLU6(input_channels, 32, 3, stride=2)
        self.stem_downsample = AMPVNetDownsample(32)
        stages: list[nn.Module] = []
        in_channels = 32
        for out_channels in self.spec.stage_channels:
            stages.append(
                AMPVNetBottleneck(
                    in_channels, out_channels, self.spec.expansion_ratio
                )
            )
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward_stages(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = [self.stem_downsample(self.stem_conv(x))]
        for stage in self.stages:
            outputs.append(stage(outputs[-1]))
        return outputs

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pool(self.forward_stages(x)[-1]).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def reconstruction_metadata(self) -> dict[str, object]:
        return {
            "source": "Luo et al., IEEE TIFS 2024, Figure 9",
            "status": self.reconstruction_status,
            "stem_channels": 32,
            "stage_channels": list(self.spec.stage_channels),
            "blocks_per_stage": [1, 1, 1, 1],
            "inferred_expansion_ratio": self.spec.expansion_ratio,
            "shortcut": False,
            "downsampling": "average_pooling",
            "dropout_assumption": self.spec.dropout,
        }


def build_ampvnet(num_classes: int, pretrained: bool = False,
                  input_channels: int = 3) -> AMPVNet:
    if pretrained:
        raise ValueError("ampvnet has no audited public pretrained weights")
    return AMPVNet(num_classes=num_classes, input_channels=input_channels)

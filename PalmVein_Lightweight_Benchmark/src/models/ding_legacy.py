from __future__ import annotations

import torch.nn as nn


class LegacyConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class LegacyDepthwisePointwiseBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride,
                1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class LegacyDingPruned(nn.Module):
    """Former parameter-matched model retained only to load archived results."""

    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        channels = [12, 24, 48, 72, 144]
        blocks = []
        in_channels_current = input_channels
        for index, out_channels in enumerate(channels):
            block = LegacyConvBlock if index == 0 else LegacyDepthwisePointwiseBlock
            blocks.append(block(in_channels_current, out_channels))
            in_channels_current = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels_current, num_classes)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


def build_ding_pruned_legacy(num_classes: int, input_channels: int = 3):
    return LegacyDingPruned(num_classes, input_channels)

from __future__ import annotations

import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, stride: int = 2):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class DepthwisePointwiseBlock(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, stride: int = 2):
        super().__init__(
            nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class DingReconstruction(nn.Module):
    """Paper-constrained independent reconstruction, not authors' source code."""

    def __init__(self, channels: list[int], num_classes: int, input_channels: int, depthwise: bool):
        super().__init__()
        blocks = []
        in_c = input_channels
        for index, out_c in enumerate(channels):
            block_cls = ConvBlock if not depthwise or index == 0 else DepthwisePointwiseBlock
            blocks.append(block_cls(in_c, out_c))
            in_c = out_c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_c, num_classes)
        self.reconstruction_status = "paper-constrained independent reconstruction"

    def forward_features(self, x):
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


def build_ding_baseline(num_classes: int, input_channels: int = 3):
    # Widths selected to reproduce the paper-level ~0.351M reference envelope
    # for one-channel input and 500 classes; topology is an explicit assumption.
    return DingReconstruction([16, 32, 64, 112, 176], num_classes, input_channels, depthwise=False)


def build_ding_pw(num_classes: int, input_channels: int = 3):
    return DingReconstruction([16, 32, 64, 128, 239], num_classes, input_channels, depthwise=True)


def build_ding_pruned(num_classes: int, input_channels: int = 3):
    return DingReconstruction([12, 24, 48, 72, 144], num_classes, input_channels, depthwise=True)

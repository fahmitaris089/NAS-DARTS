from __future__ import annotations

import torch.nn as nn


class DepthwiseSeparableBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class ChenStudentReconstruction(nn.Module):
    """Paper-constrained independent reconstruction of Chen et al.'s StudentNet."""

    reconstruction_status = "paper-constrained independent reconstruction"

    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.input_flow = nn.Sequential(
            DepthwiseSeparableBlock(32, 32, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.middle_flow = nn.Sequential(
            DepthwiseSeparableBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableBlock(64, 96),
            DepthwiseSeparableBlock(96, 128),
        )
        self.output_flow = nn.Sequential(
            DepthwiseSeparableBlock(128, 256, stride=2),
            DepthwiseSeparableBlock(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward_stages(self, x):
        x = self.stem(x)
        input_features = self.input_flow(x)
        middle_features = self.middle_flow(input_features)
        output_features = self.output_flow(middle_features)
        return input_features, middle_features, output_features

    def forward_features(self, x):
        _, _, output_features = self.forward_stages(x)
        return self.pool(output_features).flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


def build_chen_student_recon(num_classes: int, input_channels: int = 3):
    return ChenStudentReconstruction(num_classes=num_classes, input_channels=input_channels)

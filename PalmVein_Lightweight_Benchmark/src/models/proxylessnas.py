from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


OFFICIAL_CONFIG_URL = "https://raw.githubusercontent.com/han-cai/files/master/proxylessnas/proxyless_mobile.config"
OFFICIAL_WEIGHT_URL = "https://raw.githubusercontent.com/han-cai/files/master/proxylessnas/proxyless_mobile.pth"
OFFICIAL_COMMIT = "b23018c9c369d22931f7422b71ca6a7eaa354c46"

# (in, out, kernel, stride, expansion, residual); None represents an official
# ZeroLayer whose identity shortcut is the entire block.
PROXYLESS_MOBILE_BLOCKS = [
    (32, 16, 3, 1, 1, False), (16, 32, 5, 2, 3, False),
    (32, 32, 3, 1, 3, True), None, None,
    (32, 40, 7, 2, 3, False), (40, 40, 3, 1, 3, True),
    (40, 40, 5, 1, 3, True), (40, 40, 5, 1, 3, True),
    (40, 80, 7, 2, 6, False), (80, 80, 5, 1, 3, True),
    (80, 80, 5, 1, 3, True), (80, 80, 5, 1, 3, True),
    (80, 96, 5, 1, 6, False), (96, 96, 5, 1, 3, True),
    (96, 96, 5, 1, 3, True), (96, 96, 5, 1, 3, True),
    (96, 192, 7, 2, 6, False), (192, 192, 7, 1, 6, True),
    (192, 192, 7, 1, 3, True), (192, 192, 7, 1, 3, True),
    (192, 320, 7, 1, 6, False),
]


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, *, relu: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.activation = nn.ReLU6(inplace=True) if relu else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.activation(x) if self.activation is not None else x


class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class MBInvertedConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden = round(in_channels * expand_ratio)
        self.inverted_bottleneck = None
        if expand_ratio > 1:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(in_channels, hidden, 1, bias=False)),
                ("bn", nn.BatchNorm2d(hidden, eps=0.001, momentum=0.1)),
                ("relu", nn.ReLU6(inplace=True)),
            ]))
        self.depth_conv = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(hidden, hidden, kernel_size, stride, kernel_size // 2, groups=hidden, bias=False)),
            ("bn", nn.BatchNorm2d(hidden, eps=0.001, momentum=0.1)),
            ("relu", nn.ReLU6(inplace=True)),
        ]))
        self.point_linear = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(hidden, out_channels, 1, bias=False)),
            ("bn", nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        return self.point_linear(self.depth_conv(x))


class ZeroLayer(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class IdentityLayer(nn.Module):
    def forward(self, x):
        return x


class MobileInvertedResidualBlock(nn.Module):
    def __init__(self, spec):
        super().__init__()
        if spec is None:
            self.mobile_inverted_conv = ZeroLayer()
            self.shortcut = IdentityLayer()
        else:
            in_c, out_c, kernel, stride, expansion, residual = spec
            self.mobile_inverted_conv = MBInvertedConvLayer(in_c, out_c, kernel, stride, expansion)
            self.shortcut = IdentityLayer() if residual else None

    def forward(self, x):
        if isinstance(self.mobile_inverted_conv, ZeroLayer):
            return x
        value = self.mobile_inverted_conv(x)
        return value if self.shortcut is None else value + self.shortcut(x)


class ProxylessNASMobile(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.first_conv = ConvLayer(3, 32, 3, 2)
        self.blocks = nn.ModuleList(MobileInvertedResidualBlock(spec) for spec in PROXYLESS_MOBILE_BLOCKS)
        self.feature_mix_layer = ConvLayer(320, 1280, 1, 1)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = LinearLayer(1280, num_classes)

    def forward_features(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        return self.global_avg_pooling(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


def build_proxylessnas_mobile(num_classes: int, pretrained: bool = False) -> ProxylessNASMobile:
    model = ProxylessNASMobile(num_classes=1000 if pretrained else num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(OFFICIAL_WEIGHT_URL, map_location="cpu", progress=True)
        state = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state, strict=True)
        model.classifier = LinearLayer(1280, num_classes)
    return model

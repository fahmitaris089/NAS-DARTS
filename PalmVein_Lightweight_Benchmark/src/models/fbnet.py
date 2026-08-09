from __future__ import annotations

import torch.nn as nn


OFFICIAL_COMMIT = "7ed7e9177482140f58b5a56cc1acf54ecb4c1326"

# Official FBNet-C definition: op, output channels, stride, expansion.
FBNET_C_STAGES = [
    [("conv3", 16, 2, 1)],
    [("ir3", 16, 1, 1)],
    [("ir3", 24, 2, 6), ("skip", 24, 1, 1), ("ir3", 24, 1, 1), ("ir3", 24, 1, 1)],
    [("ir5", 32, 2, 6), ("ir5", 32, 1, 3), ("ir5", 32, 1, 6), ("ir3", 32, 1, 6)],
    [("ir5", 64, 2, 6), ("ir5", 64, 1, 3), ("ir5", 64, 1, 6), ("ir5", 64, 1, 6),
     ("ir5", 112, 1, 6), ("ir5", 112, 1, 6), ("ir5", 112, 1, 6), ("ir5", 112, 1, 3)],
    [("ir5", 184, 2, 6), ("ir5", 184, 1, 6), ("ir5", 184, 1, 6), ("ir5", 184, 1, 6), ("ir3", 352, 1, 6)],
    [("conv1", 1984, 1, 1)],
]


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, kernel, stride=1, groups=1, activation=True, bias=True):
        padding = kernel // 2
        layers = [nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=bias), nn.BatchNorm2d(out_c)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class FBNetIRBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int, stride: int, expansion: int):
        super().__init__()
        hidden = int(round(in_c * expansion))
        self.expand = ConvBNReLU(in_c, hidden, 1) if expansion != 1 else nn.Identity()
        self.depthwise = ConvBNReLU(hidden, hidden, kernel, stride, groups=hidden)
        self.project = ConvBNReLU(hidden, out_c, 1, activation=False)
        self.use_residual = stride == 1 and in_c == out_c

    def forward(self, x):
        value = self.project(self.depthwise(self.expand(x)))
        return x + value if self.use_residual else value


class FBNetC(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        layers = []
        in_c = 3
        for stage in FBNET_C_STAGES:
            for op, out_c, stride, expansion in stage:
                if op == "skip":
                    if in_c != out_c or stride != 1:
                        layers.append(ConvBNReLU(in_c, out_c, 1, stride, activation=False))
                elif op.startswith("conv"):
                    layers.append(ConvBNReLU(in_c, out_c, int(op[-1]), stride))
                else:
                    layers.append(FBNetIRBlock(in_c, out_c, int(op[-1]), stride, expansion))
                in_c = out_c
        self.backbone = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Conv2d(in_c, num_classes, 1)

    def forward_features(self, x):
        return self.avg_pool(self.backbone(x)).flatten(1)

    def forward(self, x):
        features = self.avg_pool(self.backbone(x))
        return self.head(features).flatten(1)


def build_fbnet_c(num_classes: int, pretrained: bool = False):
    if not pretrained:
        return FBNetC(num_classes)
    try:
        from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
    except ImportError as exc:
        raise RuntimeError(
            "Official FBNet-C pretrained weights require mobile-cv. Install requirements.txt; "
            "scratch training remains dependency-free."
        ) from exc
    model = fbnet("fbnet_c", pretrained=True)
    input_dim = model.backbone.out_channels
    model.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(input_dim, num_classes, 1), nn.Flatten(1))
    return model

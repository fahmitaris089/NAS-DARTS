from __future__ import annotations

import torch
import torch.nn as nn


MNASNET_A1_STAGES = [
    # expansion, channels, repeats, stride, kernel
    (1, 16, 1, 1, 3), (3, 24, 3, 2, 3), (3, 40, 3, 2, 5),
    (6, 80, 3, 2, 5), (6, 96, 2, 1, 3), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3),
]


class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, expansion):
        super().__init__()
        hidden = in_c * expansion
        layers = []
        if expansion != 1:
            layers += [nn.Conv2d(in_c, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, kernel, stride, kernel // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_c, 1, bias=False), nn.BatchNorm2d(out_c),
        ]
        self.layers = nn.Sequential(*layers)
        self.residual = stride == 1 and in_c == out_c

    def forward(self, x):
        value = self.layers(x)
        return x + value if self.residual else value


class MnasNetA1(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        layers = [nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)]
        in_c = 32
        for expansion, out_c, repeats, stride, kernel in MNASNET_A1_STAGES:
            for index in range(repeats):
                layers.append(InvertedResidual(in_c, out_c, kernel, stride if index == 0 else 1, expansion))
                in_c = out_c
        layers += [nn.Conv2d(in_c, 1280, 1, bias=False), nn.BatchNorm2d(1280), nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))

    def forward_features(self, x):
        return self.pool(self.layers(x)).flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


def _load_torchvision_weights_by_shape(model: MnasNetA1) -> None:
    from torchvision.models import MNASNet1_0_Weights, mnasnet1_0

    reference = mnasnet1_0(weights=MNASNet1_0_Weights.IMAGENET1K_V1)
    source = list(reference.state_dict().items())
    target = list(model.state_dict().items())
    source_shapes = [tuple(value.shape) for _, value in source]
    target_shapes = [tuple(value.shape) for _, value in target]
    if source_shapes != target_shapes:
        raise RuntimeError(
            "torchvision.mnasnet1_0 failed the explicit MnasNet-A1 tensor-shape equivalence audit; "
            "pretrained weights were not loaded."
        )
    model.load_state_dict({target_name: source_value for (target_name, _), (_, source_value) in zip(target, source)}, strict=True)


def build_mnasnet_a1(num_classes: int, pretrained: bool = False) -> MnasNetA1:
    model = MnasNetA1(num_classes=1000 if pretrained else num_classes)
    if pretrained:
        _load_torchvision_weights_by_shape(model)
        model.classifier[1] = nn.Linear(1280, num_classes)
    return model

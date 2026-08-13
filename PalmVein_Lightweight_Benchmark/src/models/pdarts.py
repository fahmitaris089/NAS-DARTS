from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def forward(self, x):
        return x


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability <= 0:
            return x
        keep = 1.0 - self.probability
        mask = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class FactorizedReduce(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        if out_c % 2:
            raise ValueError("FactorizedReduce output channels must be even")
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_c, out_c // 2, 1, 2, bias=False)
        self.conv2 = nn.Conv2d(in_c, out_c // 2, 1, 2, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.relu(x)
        return self.bn(torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1))


class DilConv(nn.Sequential):
    def __init__(self, channels: int, stride: int):
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, stride, 2, dilation=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )


class RepConvBN(nn.Module):
    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.conv_k = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn_k = nn.BatchNorm2d(channels)
        self.conv_1 = nn.Conv2d(channels, channels, 1, stride, 0, bias=False)
        self.bn_1 = nn.BatchNorm2d(channels)
        self.bn_identity = nn.BatchNorm2d(channels) if stride == 1 else None
        self.relu = nn.ReLU(inplace=True)
        self.channels = channels
        self.stride = stride
        self.deploy = False
        self.fused_conv = None

    def forward(self, x):
        if self.deploy and self.fused_conv is not None:
            return self.relu(self.fused_conv(x))
        value = self.bn_k(self.conv_k(x)) + self.bn_1(self.conv_1(x))
        if self.bn_identity is not None:
            value = value + self.bn_identity(x)
        return self.relu(value)

    @staticmethod
    def _fuse_branch(weight, bn):
        std = torch.sqrt(bn.running_var + bn.eps)
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        return weight * scale, bn.bias - bn.running_mean * bn.weight / std

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return
        weight, bias = self._fuse_branch(self.conv_k.weight, self.bn_k)
        weight_1, bias_1 = self._fuse_branch(self.conv_1.weight, self.bn_1)
        weight = weight + F.pad(weight_1, [1, 1, 1, 1])
        bias = bias + bias_1
        if self.bn_identity is not None:
            identity = torch.zeros(
                self.channels, self.channels, 1, 1,
                device=weight.device, dtype=weight.dtype,
            )
            indices = torch.arange(self.channels, device=weight.device)
            identity[indices, indices, 0, 0] = 1.0
            weight_id, bias_id = self._fuse_branch(identity, self.bn_identity)
            weight = weight + F.pad(weight_id, [1, 1, 1, 1])
            bias = bias + bias_id
        self.fused_conv = nn.Conv2d(
            self.channels, self.channels, 3, self.stride, 1, bias=True
        ).to(device=weight.device, dtype=weight.dtype)
        self.fused_conv.weight.copy_(weight)
        self.fused_conv.bias.copy_(bias)
        for name in ("conv_k", "bn_k", "conv_1", "bn_1", "bn_identity"):
            if hasattr(self, name):
                delattr(self, name)
        self.deploy = True


def fuse_reparam_model(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, RepConvBN) and not module.deploy:
            module.switch_to_deploy()
            count += 1
    return count


def make_op(name: str, channels: int, stride: int):
    if name == "skip_connect":
        return Identity() if stride == 1 else FactorizedReduce(channels, channels)
    if name == "dil_conv_3x3":
        return DilConv(channels, stride)
    if name == "rep_conv_3x3":
        return RepConvBN(channels, stride)
    raise KeyError(f"Unsupported genotype operation: {name}")


class Cell(nn.Module):
    def __init__(self, genotype_ops, in_pp, in_p, channels, reduction: bool, reduction_prev: bool):
        super().__init__()
        self.preprocess0 = FactorizedReduce(in_pp, channels) if reduction_prev else nn.Sequential(
            nn.ReLU(inplace=False), nn.Conv2d(in_pp, channels, 1, bias=False), nn.BatchNorm2d(channels)
        )
        self.preprocess1 = nn.Sequential(
            nn.ReLU(inplace=False), nn.Conv2d(in_p, channels, 1, bias=False), nn.BatchNorm2d(channels)
        )
        self.indices = []
        self.ops = nn.ModuleList()
        for name, source in genotype_ops:
            self.indices.append(int(source))
            self.ops.append(make_op(name, channels, 2 if reduction and int(source) < 2 else 1))
        self.drop_path = DropPath(0.0)

    def forward(self, s0, s1):
        states = [self.preprocess0(s0), self.preprocess1(s1)]
        for node in range(4):
            outputs = []
            for edge in range(2):
                index = node * 2 + edge
                value = self.ops[index](states[self.indices[index]])
                if not isinstance(self.ops[index], Identity):
                    value = self.drop_path(value)
                outputs.append(value)
            states.append(outputs[0] + outputs[1])
        return torch.cat(states[2:], dim=1)


class PDARTSReference(nn.Module):
    def __init__(self, genotype: dict, C_init: int, num_cells: int, num_classes: int, stem_downsample: int, reduction_indices: list[int], dropout: float = 0.0):
        super().__init__()
        stem_c = C_init * 3
        stem = [nn.Conv2d(3, stem_c, 3, 2, 1, bias=False), nn.BatchNorm2d(stem_c)]
        down_steps = max(1, stem_downsample.bit_length() - 1)
        stem += [nn.MaxPool2d(3, 2, 1) for _ in range(down_steps - 1)]
        self.stem = nn.Sequential(*stem)
        self.cells = nn.ModuleList()
        in_pp = in_p = stem_c
        reduction_prev = False
        multiplier = 1
        reductions = set(reduction_indices)
        for index in range(num_cells):
            reduction = index in reductions
            if reduction:
                multiplier *= 2
            channels = C_init * multiplier
            ops = genotype["reduce"] if reduction else genotype["normal"]
            cell = Cell(ops, in_pp, in_p, channels, reduction, reduction_prev)
            self.cells.append(cell)
            reduction_prev = reduction
            in_pp, in_p = in_p, channels * 4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_p, num_classes)

    def set_drop_path_prob(self, probability: float) -> None:
        for cell in self.cells:
            cell.drop_path.probability = float(probability)

    def forward_features(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        return self.pool(s1).flatten(1)

    def forward(self, x):
        return self.classifier(self.dropout(self.forward_features(x)))


def build_pdarts(config_path: str | Path, num_classes: int):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return PDARTSReference(
        config["genotype"], int(config["C_init"]), int(config["num_cells"]), num_classes,
        int(config["stem_downsample"]), list(config["reduction_indices"]), float(config.get("dropout", 0.0)),
    )

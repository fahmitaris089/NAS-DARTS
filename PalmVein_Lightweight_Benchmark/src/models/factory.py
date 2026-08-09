from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from src.common import PROJECT_ROOT
from .ding import build_ding_baseline, build_ding_pruned, build_ding_pw
from .fbnet import build_fbnet_c
from .mnasnet import build_mnasnet_a1
from .pdarts import build_pdarts
from .proxylessnas import build_proxylessnas_mobile


MODEL_NAMES = (
    "proxylessnas_mobile", "fbnet_c", "mnasnet_a1", "ding_baseline",
    "ding_pw", "ding_pruned", "pdarts_l005_c12_cells10",
)
PRETRAINED_MODELS = {"proxylessnas_mobile", "fbnet_c", "mnasnet_a1"}


def build_model(name: str, num_classes: int = 834, *, pretrained: bool = False, input_channels: int = 3):
    if name not in MODEL_NAMES:
        raise KeyError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}")
    if pretrained and name not in PRETRAINED_MODELS:
        raise ValueError(f"{name} has no official pretrained weights; pretrained protocol is N/A")
    if name == "proxylessnas_mobile":
        return build_proxylessnas_mobile(num_classes, pretrained)
    if name == "fbnet_c":
        return build_fbnet_c(num_classes, pretrained)
    if name == "mnasnet_a1":
        return build_mnasnet_a1(num_classes, pretrained)
    if name == "ding_baseline":
        return build_ding_baseline(num_classes, input_channels)
    if name == "ding_pw":
        return build_ding_pw(num_classes, input_channels)
    if name == "ding_pruned":
        return build_ding_pruned(num_classes, input_channels)
    return build_pdarts(PROJECT_ROOT / "configs/models/pdarts_l005_c12_cells10.json", num_classes)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad or not trainable_only)


def get_classifier_parameters(model: nn.Module):
    for name in ("classifier", "head"):
        module = getattr(model, name, None)
        if module is not None:
            return list(module.parameters())
    raise AttributeError(f"Classifier module not found in {type(model).__name__}")

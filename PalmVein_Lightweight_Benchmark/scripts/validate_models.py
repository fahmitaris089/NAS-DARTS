#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MODEL_NAMES, build_model, count_parameters
from src.models.ding import (
    DING_BASELINE_SPECS,
    DING_PRUNED_SPECS,
    DING_PW_SPECS,
    DingReconstruction,
)
from src.models.factory import PALMNET_VARIANTS
from src.models.palmnet import PalmNet, build_palmnet


DING_SPECS = {
    "ding_baseline": DING_BASELINE_SPECS,
    "ding_pw": DING_PW_SPECS,
    "ding_pruned": DING_PRUNED_SPECS,
}

EXACT_REFERENCE_TARGETS = {
    "mnasnet_a1": 3_887_038,
    "mnasnet_b1_torchvision": 4_383_312,
}

PALMNET_PAPER_DIAGNOSTICS = {
    "palmnet_05x_2223": (0.56, 37.67),
    "palmnet_05x_4223": (0.56, 40.11),
    "palmnet_05x_6223": (0.56, 42.56),
    "palmnet_05x_2323": (0.58, 43.09),
    "palmnet_05x_2313": (0.46, 37.98),
    "palmnet_05x_2413": (0.49, 43.40),
    "palmnet_05x_2412": (0.44, 34.11),
    "palmnet_05x_2411": (0.39, 24.83),
    "palmnet_10x_2413": (1.37, 192.77),
    "palmnet_20x_2413": (4.47, 765.12),
}


def profile_mmacs(model, input_channels: int = 3) -> float | None:
    try:
        from thop import profile
    except ImportError:
        return None
    model.eval()
    macs, _ = profile(model, inputs=(torch.zeros(1, input_channels, 224, 224),), verbose=False)
    return float(macs / 1_000_000)


def validate_ding_architecture(name: str, model: DingReconstruction) -> None:
    expected = DING_SPECS[name]
    if len(model.blocks) != 6 or tuple(model.architecture_spec[1:]) != tuple(expected[1:]):
        raise RuntimeError(f"{name} does not match the six-block paper-constrained specification")
    if model.architecture_spec[0].out_channels != expected[0].out_channels:
        raise RuntimeError(f"{name} first block output differs from the paper specification")
    grouped = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d) and module.groups != 1]
    if grouped:
        raise RuntimeError(f"{name} unexpectedly contains grouped/depthwise convolutions")
    with torch.inference_mode():
        features = model.forward_block_features(torch.zeros(1, 3, 224, 224))
    spatial = [tuple(value.shape[-2:]) for value in features]
    if spatial != [(112, 112), (56, 56), (28, 28), (14, 14), (7, 7), (7, 7)]:
        raise RuntimeError(f"{name} feature-map progression is invalid: {spatial}")


def validate_palmnet_architecture(name: str, model: PalmNet) -> None:
    width, code = PALMNET_VARIANTS[name]
    expected_counts = tuple(int(value) for value in code)
    observed_counts = (
        len(model.shuffle_stage),
        len(model.mobilenetv3_stage),
        len(model.mbconv_stage),
        model.spec.expansion_factor,
    )
    if model.spec.width_mult != width or observed_counts != expected_counts:
        raise RuntimeError(f"{name} has invalid PalmNet metadata: {observed_counts}")
    with torch.inference_mode():
        features = model.forward_stages(torch.zeros(1, 3, 224, 224))
    spatial = [tuple(value.shape[-2:]) for value in features]
    if spatial != [(56, 56), (28, 28), (14, 14), (7, 7), (7, 7)]:
        raise RuntimeError(f"{name} feature-map progression is invalid: {spatial}")


def main():
    parser = argparse.ArgumentParser(description="Validate model specifications before training")
    parser.add_argument("--skip-flops", action="store_true")
    args = parser.parse_args()
    rows = []
    for name in MODEL_NAMES:
        model = build_model(name, num_classes=834, pretrained=False)
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(2, 3, 224, 224))
        if tuple(output.shape) != (2, 834):
            raise RuntimeError(f"{name} output is {tuple(output.shape)}, expected (2, 834)")
        row = {
            "model": name,
            "adapted_input_channels": 3,
            "adapted_num_classes": 834,
            "adapted_parameters": count_parameters(model),
            "mmacs_224": None if args.skip_flops else profile_mmacs(model),
            "output_shape_valid": True,
            "reference_parameters": "",
            "reference_delta_fraction": "",
            "reference_valid": "N/A",
            "reference_mmacs_224": "",
            "paper_reported_parameters_m": "",
            "paper_reported_flops_m": "",
            "provenance_status": "",
        }
        if name in DING_SPECS:
            validate_ding_architecture(name, model)
            reference = build_model(name, num_classes=500, input_channels=1)
            observed = count_parameters(reference)
            with torch.inference_mode():
                reference_output = reference(torch.zeros(1, 1, 224, 224))
            row.update(
                reference_parameters=observed,
                reference_delta_fraction="not_used_as_acceptance_criterion",
                reference_valid=tuple(reference_output.shape) == (1, 500),
            )
            if tuple(reference_output.shape) != (1, 500):
                raise RuntimeError(f"{name} paper-reference output is {tuple(reference_output.shape)}")
        elif name in EXACT_REFERENCE_TARGETS:
            target = EXACT_REFERENCE_TARGETS[name]
            observed = count_parameters(build_model(name, num_classes=1000, input_channels=3))
            row.update(
                reference_parameters=observed,
                reference_delta_fraction=abs(observed - target) / target,
                reference_valid=observed == target,
            )
            if observed != target:
                raise RuntimeError(f"{name} reference parameters {observed} do not match {target}")
        elif name in PALMNET_VARIANTS:
            validate_palmnet_architecture(name, model)
            width, code = PALMNET_VARIANTS[name]
            reference = build_palmnet(
                width_mult=width,
                variant_code=code,
                num_classes=200,
                input_channels=1,
            ).eval()
            with torch.inference_mode():
                reference_output = reference(torch.zeros(1, 1, 224, 224))
            reported_parameters, reported_flops = PALMNET_PAPER_DIAGNOSTICS[name]
            row.update(
                reference_parameters=count_parameters(reference),
                reference_delta_fraction="diagnostic_only_missing_layer_specification",
                reference_valid=tuple(reference_output.shape) == (1, 200),
                reference_mmacs_224=None if args.skip_flops else profile_mmacs(reference, 1),
                paper_reported_parameters_m=reported_parameters,
                paper_reported_flops_m=reported_flops,
                provenance_status="paper-constrained independent reconstruction",
            )
            if tuple(reference_output.shape) != (1, 200):
                raise RuntimeError(f"{name} reference output is {tuple(reference_output.shape)}")
        rows.append(row)
        print(json.dumps(row), flush=True)
    palmnet_rows = {row["model"]: row for row in rows if row["model"] in PALMNET_VARIANTS}
    compact = palmnet_rows["palmnet_05x_2411"]
    standard = palmnet_rows["palmnet_05x_2413"]
    if compact["adapted_parameters"] >= standard["adapted_parameters"]:
        raise RuntimeError("PalmNet-0.5x2411 must have fewer parameters than PalmNet-0.5x2413")
    if not args.skip_flops and compact["mmacs_224"] >= standard["mmacs_224"]:
        raise RuntimeError("PalmNet-0.5x2411 must have fewer MMACs than PalmNet-0.5x2413")
    output = PROJECT_ROOT / "results/model_validation/model_spec_validation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()

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


REFERENCE_TARGETS = {
    "ding_baseline": (351_000, 0.03),
    "ding_pw": (165_000, 0.03),
    "ding_pruned": (93_000, 0.05),
}


def profile_mmacs(model) -> float | None:
    try:
        from thop import profile
    except ImportError:
        return None
    model.eval()
    macs, _ = profile(model, inputs=(torch.zeros(1, 3, 224, 224),), verbose=False)
    return float(macs / 1_000_000)


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
        }
        if name in REFERENCE_TARGETS:
            target, tolerance = REFERENCE_TARGETS[name]
            reference = build_model(name, num_classes=500, input_channels=1)
            observed = count_parameters(reference)
            delta = abs(observed - target) / target
            row.update(reference_parameters=observed, reference_delta_fraction=delta, reference_valid=delta <= tolerance)
            if delta > tolerance:
                raise RuntimeError(f"{name} reference parameters {observed} outside {tolerance:.1%} of {target}")
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = PROJECT_ROOT / "results/model_validation/model_spec_validation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()

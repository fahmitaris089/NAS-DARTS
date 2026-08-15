#!/usr/bin/env python3
"""Validation-only checkpoint error and overlap audit.

This command deliberately builds only the validation dataset. It never creates
or evaluates a test loader, so its outputs are safe for screening diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_center_relation import sha256_file, stable_json_hash  # noqa: E402
from kd_config import KDConfig  # noqa: E402
from kd_train import load_student, load_teacher  # noqa: E402
from palm_vein_dataset import (  # noqa: E402
    PalmVeinDataset, build_image_list, build_label_map, get_transforms, load_split,
)


def _logger():
    logger = logging.getLogger("validation_error_audit")
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    return logger


@torch.no_grad()
def predict(model, loader, samples, device):
    model.eval()
    rows = []
    cursor = 0
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        probabilities = torch.softmax(logits, dim=1)
        values, indices = probabilities.topk(2, dim=1)
        true_logits = logits.gather(1, labels.to(device)[:, None]).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, labels.to(device)[:, None], float("-inf"))
        margins = true_logits - masked.max(1).values
        for i in range(labels.numel()):
            path, identity = samples[cursor + i]
            rows.append({
                "sample_id": str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else str(path),
                "identity": int(identity),
                "true_class": int(labels[i]),
                "top1_prediction": int(indices[i, 0]),
                "top2_prediction": int(indices[i, 1]),
                "confidence": float(values[i, 0]),
                "true_logit_margin": float(margins[i]),
                "correct": bool(indices[i, 0].cpu() == labels[i]),
            })
        cursor += labels.numel()
    if cursor != len(samples):
        raise RuntimeError("Prediction/sample alignment failed")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    logger = _logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_path = Path(config["split_path"])
    split = load_split(split_path)
    label_map = build_label_map(split["subjects"])
    if len(label_map) != int(config["num_classes"]):
        raise ValueError("Configured class count and split label map differ")
    # Validation only: no call to create_retrain_dataloaders and no test dataset.
    samples = build_image_list(config["data_dir"], split["val"], label_map)
    dataset = PalmVeinDataset(samples, get_transforms("val", int(config["input_size"])))
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False,
                        num_workers=int(config["num_workers"]))

    base_cfg = KDConfig(
        teacher_arch=config["teacher"]["arch"],
        teacher_weights=config["teacher"]["weights"],
        num_classes=int(config["num_classes"]),
    )
    teacher = load_teacher(base_cfg, device, logger)
    teacher_rows = predict(teacher, loader, samples, device)
    teacher_predictions = {row["sample_id"]: row["top1_prediction"] for row in teacher_rows}

    model_rows = {config["teacher"]["name"]: teacher_rows}
    manifests = [{
        "name": config["teacher"]["name"], "kind": "teacher",
        "checkpoint": config["teacher"]["weights"],
        "checkpoint_sha256": sha256_file(config["teacher"]["weights"]),
    }]
    for entry in config["students"]:
        config_file, weights = Path(entry["config"]), Path(entry["weights"])
        if not config_file.exists() or not weights.exists():
            if entry.get("optional", True):
                logger.warning(f"Skipping unavailable optional checkpoint: {entry['name']}")
                continue
            raise FileNotFoundError(f"Missing checkpoint/config for {entry['name']}")
        student_cfg = KDConfig(
            student_config_path=str(config_file), student_weights=str(weights),
            num_classes=int(config["num_classes"]), no_pretrained_student=False,
        )
        student = load_student(student_cfg, device, logger)
        rows = predict(student, loader, samples, device)
        model_rows[entry["name"]] = rows
        manifests.append({
            "name": entry["name"], "kind": "student", "config": str(config_file),
            "config_sha256": sha256_file(config_file), "checkpoint": str(weights),
            "checkpoint_sha256": sha256_file(weights),
        })
        del student

    prediction_fields = [
        "model", "sample_id", "identity", "true_class", "top1_prediction",
        "top2_prediction", "confidence", "true_logit_margin",
        "teacher_prediction", "correct",
    ]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for name, rows in model_rows.items():
            for row in rows:
                writer.writerow({"model": name, **row,
                                 "teacher_prediction": teacher_predictions[row["sample_id"]]})

    error_sets = {
        name: {row["sample_id"] for row in rows if not row["correct"]}
        for name, rows in model_rows.items()
    }
    summary = []
    for item in manifests:
        errors = error_sets[item["name"]]
        summary.append({**item, "validation_samples": len(samples),
                        "correct": len(samples) - len(errors), "errors": len(errors),
                        "accuracy": (len(samples) - len(errors)) / len(samples)})
    fields = sorted({key for row in summary for key in row})
    with (output / "checkpoint_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(summary)

    names = list(error_sets)
    overlap = {}
    matrix = []
    for left in names:
        matrix_row = {"model": left}
        for right in names:
            intersection = error_sets[left] & error_sets[right]
            union = error_sets[left] | error_sets[right]
            jaccard = len(intersection) / len(union) if union else 1.0
            matrix_row[right] = jaccard
            overlap[f"{left}__{right}"] = {
                "intersection": len(intersection), "union": len(union),
                "jaccard": jaccard, "shared_errors": sorted(intersection),
            }
        matrix.append(matrix_row)
    (output / "error_overlap.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    with (output / "overlap_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *names])
        writer.writeheader(); writer.writerows(matrix)

    manifest = {
        "partition": "validation_only", "test_loader_created": False,
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "split": str(split_path), "split_sha256": sha256_file(split_path),
        "label_map_sha256": stable_json_hash(label_map),
        "preprocessing_sha256": stable_json_hash({
            "input_size": config["input_size"],
            "transform": "deterministic_validation_imagenet_normalization",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "grayscale_to_rgb": "repeat_channels",
        }),
        "models": manifests, "samples": len(samples),
    }
    (output / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "models": names, "samples": len(samples)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validation-only C10 error forensics and deterministic decision gate.

Only the training partition (for prototypes) and validation partition (for
screening) are constructed. The held-out test partition is never loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_center_relation import sha256_file, stable_json_hash  # noqa: E402
from kd_config import KDConfig  # noqa: E402
from kd_train import load_student, load_teacher, teacher_forward_with_embeddings  # noqa: E402
from palm_vein_dataset import (  # noqa: E402
    PalmVeinDataset, build_image_list, build_label_map, get_transforms, load_split,
)


def _logger() -> logging.Logger:
    logger = logging.getLogger("c10_error_forensics")
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    return logger


def _json_vector(tensor: torch.Tensor) -> str:
    return json.dumps([round(float(value), 7) for value in tensor.detach().cpu()])


def _forward(model, images: torch.Tensor, teacher_arch: str | None):
    if teacher_arch is not None:
        return teacher_forward_with_embeddings(model, teacher_arch, images)
    if not hasattr(model, "forward_with_embeddings"):
        raise TypeError("Student model must expose forward_with_embeddings()")
    return model.forward_with_embeddings(images)


@torch.no_grad()
def build_prototypes(model, loader, device, num_classes: int, teacher_arch=None):
    model.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        _, embeddings = _forward(model, images, teacher_arch)
        normalized = F.normalize(embeddings.float(), dim=1)
        if sums is None:
            sums = torch.zeros(num_classes, normalized.shape[1], device=device)
        sums.index_add_(0, labels, normalized)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.long))
    if sums is None or (counts == 0).any():
        missing = torch.where(counts == 0)[0].cpu().tolist()
        raise ValueError(f"Cannot build prototypes; missing training classes: {missing}")
    return F.normalize(sums / counts[:, None], dim=1), counts


@torch.no_grad()
def predict(model, loader, samples, prototypes, device, teacher_arch=None):
    model.eval()
    rows, cursor = [], 0
    for images, labels in loader:
        images, labels_device = images.to(device), labels.to(device)
        logits, embeddings = _forward(model, images, teacher_arch)
        probabilities = torch.softmax(logits, dim=1)
        values, indices = probabilities.topk(5, dim=1)
        true_logits = logits.gather(1, labels_device[:, None]).squeeze(1)
        true_ranks = logits.gt(true_logits[:, None]).sum(1) + 1
        masked = logits.clone()
        masked.scatter_(1, labels_device[:, None], float("-inf"))
        margins = true_logits - masked.max(1).values
        raw_norms = embeddings.float().norm(dim=1)
        normalized = F.normalize(embeddings.float(), dim=1)
        predicted = indices[:, 0]
        cosine_true = (normalized * prototypes[labels_device]).sum(1)
        cosine_pred = (normalized * prototypes[predicted]).sum(1)
        for index in range(labels.numel()):
            path, identity = samples[cursor + index]
            rows.append({
                "sample_id": str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else str(path),
                "identity": int(identity), "true_class": int(labels[index]),
                "prediction": int(predicted[index]),
                "correct": bool(predicted[index].cpu() == labels[index]),
                "true_rank": int(true_ranks[index]),
                "top1_prediction": int(indices[index, 0]),
                "top2_prediction": int(indices[index, 1]),
                "top5_predictions": json.dumps(indices[index].cpu().tolist()),
                "top5_scores": _json_vector(values[index]),
                "confidence": float(values[index, 0]),
                "true_logit": float(true_logits[index]),
                "true_class_margin": float(margins[index]),
                "raw_feature_norm": float(raw_norms[index]),
                "normalized_embedding": _json_vector(normalized[index]),
                "cosine_to_true_prototype": float(cosine_true[index]),
                "cosine_to_predicted_prototype": float(cosine_pred[index]),
            })
        cursor += labels.numel()
    if cursor != len(samples):
        raise RuntimeError("Prediction/sample alignment failed")
    norms = torch.tensor([row["raw_feature_norm"] for row in rows])
    mean = float(norms.mean())
    std = float(norms.std(unbiased=False).clamp_min(1e-12))
    for row in rows:
        row["feature_norm_z"] = (row["raw_feature_norm"] - mean) / std
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
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
    num_classes = int(config["num_classes"])
    if len(label_map) != num_classes:
        raise ValueError("Configured class count and split label map differ")

    train_samples = build_image_list(config["data_dir"], split["train"], label_map)
    val_samples = build_image_list(config["data_dir"], split["val"], label_map)
    train_dataset = PalmVeinDataset(train_samples, get_transforms("val", int(config["input_size"])))
    val_dataset = PalmVeinDataset(val_samples, get_transforms("val", int(config["input_size"])))
    kwargs = {"batch_size": int(config["batch_size"]), "shuffle": False,
              "num_workers": int(config["num_workers"])}
    train_loader, val_loader = DataLoader(train_dataset, **kwargs), DataLoader(val_dataset, **kwargs)

    teacher_cfg = KDConfig(teacher_arch=config["teacher"]["arch"],
                           teacher_weights=config["teacher"]["weights"],
                           num_classes=num_classes)
    teacher = load_teacher(teacher_cfg, device, logger)
    teacher_prototypes, teacher_counts = build_prototypes(
        teacher, train_loader, device, num_classes, config["teacher"]["arch"]
    )
    teacher_name = config["teacher"]["name"]
    model_rows = {teacher_name: predict(
        teacher, val_loader, val_samples, teacher_prototypes, device,
        config["teacher"]["arch"],
    )}
    manifests = [{"name": teacher_name, "kind": "teacher",
                  "checkpoint": config["teacher"]["weights"],
                  "checkpoint_sha256": sha256_file(config["teacher"]["weights"]),
                  "prototype_training_samples": int(teacher_counts.sum())}]

    for entry in config["students"]:
        config_file, weights = Path(entry["config"]), Path(entry["weights"])
        if not config_file.exists() or not weights.exists():
            raise FileNotFoundError(f"Missing checkpoint/config for {entry['name']}")
        student_cfg = KDConfig(student_config_path=str(config_file),
                               student_weights=str(weights), num_classes=num_classes)
        screening_file = weights.parent / "screening_results.json"
        if not screening_file.exists():
            raise FileNotFoundError(f"Missing validation screening provenance: {screening_file}")
        screening = json.loads(screening_file.read_text(encoding="utf-8"))
        observed_epoch = screening.get(
            "best_screening_epoch", screening.get("best_validation_accuracy_epoch")
        )
        if observed_epoch != entry.get("expected_epoch"):
            raise ValueError(
                f"Checkpoint epoch mismatch for {entry['name']}: "
                f"expected {entry.get('expected_epoch')}, screening records {observed_epoch}"
            )
        student = load_student(student_cfg, device, logger)
        prototypes, counts = build_prototypes(student, train_loader, device, num_classes)
        model_rows[entry["name"]] = predict(student, val_loader, val_samples,
                                             prototypes, device)
        manifests.append({"name": entry["name"], "kind": "student",
                          "expected_epoch": entry.get("expected_epoch"),
                          "observed_screening_epoch": observed_epoch,
                          "screening_epoch_source": (
                              "best_screening_epoch" if "best_screening_epoch" in screening
                              else "legacy_best_validation_accuracy_epoch"
                          ),
                          "screening_sha256": sha256_file(screening_file),
                          "config": str(config_file), "config_sha256": sha256_file(config_file),
                          "training_config": entry.get("training_config", str(config_file)),
                          "training_config_sha256": sha256_file(
                              entry.get("training_config", config_file)
                          ),
                          "checkpoint": str(weights), "checkpoint_sha256": sha256_file(weights),
                          "prototype_training_samples": int(counts.sum())})
        del student

    teacher_by_id = {row["sample_id"]: row for row in model_rows[teacher_name]}
    long_rows = []
    for name, rows in model_rows.items():
        for row in rows:
            teacher_row = teacher_by_id[row["sample_id"]]
            long_rows.append({"model": name, **row,
                              "teacher_prediction": teacher_row["prediction"],
                              "teacher_confidence": teacher_row["confidence"],
                              "teacher_correct": teacher_row["correct"]})
    _write_csv(output / "predictions.csv", long_rows)

    error_sets = {name: {row["sample_id"] for row in rows if not row["correct"]}
                  for name, rows in model_rows.items()}
    summaries = []
    for manifest in manifests:
        errors = error_sets[manifest["name"]]
        summaries.append({**manifest, "validation_samples": len(val_samples),
                          "correct": len(val_samples) - len(errors), "errors": len(errors),
                          "accuracy": (len(val_samples) - len(errors)) / len(val_samples)})
    _write_csv(output / "checkpoint_summary.csv", summaries)

    names, overlap, matrix = list(error_sets), {}, []
    for left in names:
        matrix_row = {"model": left}
        for right in names:
            intersection, union = error_sets[left] & error_sets[right], error_sets[left] | error_sets[right]
            jaccard = len(intersection) / len(union) if union else 1.0
            matrix_row[right] = jaccard
            overlap[f"{left}__{right}"] = {"intersection": len(intersection),
                "union": len(union), "jaccard": jaccard,
                "shared_errors": sorted(intersection)}
        matrix.append(matrix_row)
    (output / "error_overlap.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    _write_csv(output / "overlap_matrix.csv", matrix)

    student_names = [entry["name"] for entry in config["students"]]
    common_errors = set.intersection(*(error_sets[name] for name in student_names))
    union_errors = set.union(*(error_sets[name] for name in student_names))
    z_limit = float(config.get("feature_norm_z_threshold", 2.5))
    gap_limit = float(config.get("prototype_similarity_gap", 0.02))
    difficult = []
    for sample_id in sorted(union_errors):
        for name, rows in model_rows.items():
            row = next(item for item in rows if item["sample_id"] == sample_id)
            gap = row["cosine_to_true_prototype"] - row["cosine_to_predicted_prototype"]
            difficult.append({"model": name, **row, "prototype_similarity_gap": gap,
                              "feature_norm_extreme": abs(row["feature_norm_z"]) >= z_limit,
                              "prototype_ambiguous": abs(gap) <= gap_limit,
                              "common_student_error": sample_id in common_errors,
                              "teacher_correct": teacher_by_id[sample_id]["correct"]})
    _write_csv(output / "difficult_samples.csv", difficult)

    teacher_correct = sorted(x for x in common_errors if teacher_by_id[x]["correct"])
    teacher_wrong = sorted(common_errors - set(teacher_correct))
    student_difficult = [row for row in difficult if row["model"] != teacher_name]
    norm_extreme = any(row["feature_norm_extreme"] for row in student_difficult)
    ambiguous = any(row["prototype_ambiguous"] for row in student_difficult)
    complementary = len(union_errors) > len(common_errors)
    if teacher_correct:
        recommended = "progressive_center_relation"
    elif norm_extreme:
        recommended = "adaface_matched_control"
    elif ambiguous:
        recommended = "arcface_vs_subcenter"
    elif complementary:
        recommended = "ensemble_and_weight_soup_diagnostic"
    elif teacher_wrong:
        recommended = "data_roi_label_audit"
    else:
        recommended = "stop_no_supported_branch"
    decision = {"recommended_branch": recommended, "student_models": student_names,
                "common_student_errors": sorted(common_errors),
                "union_student_errors": sorted(union_errors),
                "teacher_correct_on_common_errors": teacher_correct,
                "teacher_wrong_on_common_errors": teacher_wrong,
                "teacher_wrong_samples_require_roi_label_audit": bool(teacher_wrong),
                "feature_norm_extreme_present": norm_extreme,
                "prototype_ambiguity_present": ambiguous,
                "student_errors_complementary": complementary,
                "test_partition_inspected": False}
    (output / "decision_gate.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    manifest = {"partition": "validation_only", "test_loader_created": False,
                "prototype_partition": "training_only", "config": str(config_path),
                "config_sha256": sha256_file(config_path), "split": str(split_path),
                "split_sha256": sha256_file(split_path),
                "label_map_sha256": stable_json_hash(label_map),
                "preprocessing_sha256": stable_json_hash({
                    "input_size": config["input_size"],
                    "transform": "deterministic_validation_imagenet_normalization",
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                    "grayscale_to_rgb": "repeat_channels"}),
                "models": manifests, "validation_samples": len(val_samples),
                "prototype_training_samples": len(train_samples), "decision_gate": decision}
    (output / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "models": names,
                      "validation_samples": len(val_samples),
                      "recommended_branch": recommended}, indent=2))


if __name__ == "__main__":
    main()

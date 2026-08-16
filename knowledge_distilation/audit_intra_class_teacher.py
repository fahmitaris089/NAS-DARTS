#!/usr/bin/env python3
"""Train/validation-only geometry audit for the fixed C20 teacher assistant."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
NAS = ROOT / "Eksperimen_Hardware_Aware_PDARTS" / "src" / "nas"
sys.path[:0] = [str(ROOT), str(NAS)]

from adaface import replace_linear_with_arcface  # noqa: E402
from genotypes import dict_to_genotype  # noqa: E402
from model_eval import EvalNetwork  # noqa: E402
from palm_vein_dataset import (  # noqa: E402
    PalmVeinDataset,
    build_image_list,
    build_label_map,
    get_transforms,
    load_split,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_reductions(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        return [int(value.strip()) for value in raw.split(",") if value.strip()]
    return [int(value) for value in raw]


def build_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("loss_mode") not in {"arcface", "subcenter_arcface"}:
        raise ValueError(f"Expected ArcFace config: {config_path}")
    model = EvalNetwork(
        genotype=dict_to_genotype(config["genotype"]),
        C_init=int(config["C_init"]),
        num_cells=int(config["num_cells"]),
        num_classes=834,
        auxiliary=False,
        dropout=float(config.get("student_dropout", 0.3)),
        stem_downsample=int(config.get("stem_downsample", 8)),
        reduction_indices=parse_reductions(config.get("reduction_indices")),
    )
    mode = config["loss_mode"]
    replace_linear_with_arcface(
        model,
        num_classes=834,
        m=float(config.get("arcface_margin", 0.5)),
        s=float(config.get("arcface_scale", 64.0)),
        num_subcenters=int(config.get(
            "arcface_subcenters", 2 if mode == "subcenter_arcface" else 1
        )),
        margin_warmup_epochs=int(config.get("arcface_margin_warmup_epochs", 0)),
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "student" in state:
        state = state["student"]
    state = {key: value for key, value in state.items()
             if not key.startswith("_auxiliary_head")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    material_missing = [key for key in missing if not key.startswith("_auxiliary_head")]
    if material_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/config mismatch for {checkpoint_path}: "
            f"missing={material_missing[:10]} unexpected={unexpected[:10]}"
        )
    return model.to(device).eval(), config


@torch.no_grad()
def extract(model, samples, *, batch_size, num_workers, device):
    dataset = PalmVeinDataset(samples, get_transforms("val", 224))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    embeddings, logits, labels = [], [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        output, feature = model.forward_with_embeddings(images)
        embeddings.append(F.normalize(feature.float(), dim=1).cpu())
        logits.append(output.float().cpu())
        labels.append(target.long())
    return torch.cat(embeddings), torch.cat(logits), torch.cat(labels)


def summarize(train_embeddings, train_labels, val_embeddings, val_logits,
              val_labels, num_classes):
    sums = torch.zeros(num_classes, train_embeddings.shape[1], dtype=torch.float64)
    counts = torch.zeros(num_classes, dtype=torch.long)
    sums.index_add_(0, train_labels, train_embeddings.double())
    counts.index_add_(0, train_labels, torch.ones_like(train_labels))
    if bool((counts == 0).any()):
        raise ValueError("A class is missing from the training partition")
    prototypes = F.normalize((sums / counts[:, None]).float(), dim=1)
    similarities = val_embeddings @ prototypes.T
    true_similarity = similarities.gather(1, val_labels[:, None]).squeeze(1)
    rivals = similarities.clone()
    rivals.scatter_(1, val_labels[:, None], float("-inf"))
    rival_similarity, nearest_wrong = rivals.max(dim=1)
    centroid_prediction = similarities.argmax(dim=1)
    model_prediction = val_logits.argmax(dim=1)
    return {
        "nearest_centroid_errors": int((centroid_prediction != val_labels).sum()),
        "classifier_errors": int((model_prediction != val_labels).sum()),
        "median_true_centroid_cosine": float(true_similarity.median()),
        "p05_true_centroid_cosine": float(torch.quantile(true_similarity, 0.05)),
        "mean_true_centroid_cosine": float(true_similarity.mean()),
        "mean_nearest_wrong_cosine": float(rival_similarity.mean()),
        "mean_centroid_margin": float((true_similarity - rival_similarity).mean()),
        "mean_intra_class_spread": float((1.0 - true_similarity).mean()),
        "per_sample": {
            "true_similarity": true_similarity,
            "rival_similarity": rival_similarity,
            "nearest_wrong": nearest_wrong,
            "centroid_prediction": centroid_prediction,
            "model_prediction": model_prediction,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--student-config", required=True, type=Path)
    parser.add_argument("--student-checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--split-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split(args.split_path)
    label_map = build_label_map(split["subjects"])
    train_samples = build_image_list(args.data_dir, split["train"], label_map)
    val_samples = build_image_list(args.data_dir, split["val"], label_map)
    if len(label_map) != 834:
        raise ValueError(f"Expected 834 classes, found {len(label_map)}")

    results = {}
    sample_metrics = {}
    for name, config, checkpoint in (
        ("teacher_c20", args.teacher_config, args.teacher_checkpoint),
        ("student_c10", args.student_config, args.student_checkpoint),
    ):
        model, _ = build_model(config, checkpoint, device)
        train_emb, _, train_labels = extract(
            model, train_samples, batch_size=args.batch_size,
            num_workers=args.num_workers, device=device,
        )
        val_emb, val_logits, val_labels = extract(
            model, val_samples, batch_size=args.batch_size,
            num_workers=args.num_workers, device=device,
        )
        summary = summarize(
            train_emb, train_labels, val_emb, val_logits, val_labels, len(label_map)
        )
        sample_metrics[name] = summary.pop("per_sample")
        results[name] = summary
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    teacher = results["teacher_c20"]
    student = results["student_c10"]
    teacher_errors = set(torch.where(
        sample_metrics["teacher_c20"]["model_prediction"]
        != torch.tensor([label for _, label in val_samples])
    )[0].tolist())
    student_errors = set(torch.where(
        sample_metrics["student_c10"]["model_prediction"]
        != torch.tensor([label for _, label in val_samples])
    )[0].tolist())
    overlap = teacher_errors & student_errors
    union = teacher_errors | student_errors
    error_overlap = {
        "teacher_error_count": len(teacher_errors),
        "student_error_count": len(student_errors),
        "intersection_count": len(overlap),
        "union_count": len(union),
        "jaccard": float(len(overlap) / len(union)) if union else 1.0,
        "shared_error_samples": [str(val_samples[index][0]) for index in sorted(overlap)],
    }
    geometry_improved = (
        teacher["median_true_centroid_cosine"] > student["median_true_centroid_cosine"]
        or teacher["p05_true_centroid_cosine"] > student["p05_true_centroid_cosine"]
    )
    gate_pass = (
        teacher["nearest_centroid_errors"] <= student["nearest_centroid_errors"]
        and geometry_improved
    )
    decision = {
        "status": "PASS" if gate_pass else "FAIL",
        "rule": (
            "teacher nearest-centroid errors <= student AND teacher improves "
            "median or p05 true-centroid cosine"
        ),
        "test_loader_created": False,
        "test_partition_inspected": False,
        "teacher": teacher,
        "student": student,
        "classifier_error_overlap": error_overlap,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (path, label) in enumerate(val_samples):
        row = {"sample_id": str(path), "true_class": int(label)}
        for name in ("teacher_c20", "student_c10"):
            metrics = sample_metrics[name]
            row.update({
                f"{name}_prediction": int(metrics["model_prediction"][index]),
                f"{name}_centroid_prediction": int(metrics["centroid_prediction"][index]),
                f"{name}_true_centroid_cosine": float(metrics["true_similarity"][index]),
                f"{name}_nearest_wrong_cosine": float(metrics["rival_similarity"][index]),
                f"{name}_nearest_wrong_class": int(metrics["nearest_wrong"][index]),
            })
        rows.append(row)
    with (args.output_dir / "per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "teacher_audit.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    manifest = {
        "task": "closed_set_identification",
        "partitions_used": ["train", "val"],
        "test_loader_created": False,
        "split_sha256": sha256_file(args.split_path),
        "teacher_config_sha256": sha256_file(args.teacher_config),
        "teacher_checkpoint_sha256": sha256_file(args.teacher_checkpoint),
        "student_config_sha256": sha256_file(args.student_config),
        "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
        "preprocessing": "deterministic validation transform at 224x224",
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
    }
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

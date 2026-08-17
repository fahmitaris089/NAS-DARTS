#!/usr/bin/env python3
"""Build and evaluate a frozen class-prototype identification head.

Class prototypes are constructed exclusively from the training partition using
deterministic evaluation transforms.  Validation and test images never update
the prototypes.  Test evaluation requires an already-built prototype artifact
and an explicit acknowledgement because this project's test split has already
been observed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
NAS = ROOT / "Eksperimen_Hardware_Aware_PDARTS" / "src" / "nas"
sys.path[:0] = [str(ROOT), str(NAS), str(ROOT / "scripts")]

from evaluate_frozen_identification import build_model, sha  # noqa: E402
from palm_vein_dataset import (  # noqa: E402
    PalmVeinDataset,
    build_image_list,
    build_label_map,
    get_transforms,
    load_split,
)


def sample_manifest_sha(samples: list[tuple[Path, int]], data_dir: Path) -> str:
    """Hash the ordered relative paths and labels without reading image bytes."""
    records = []
    for path, label in samples:
        try:
            relative = path.resolve().relative_to(data_dir.resolve()).as_posix()
        except ValueError:
            relative = path.resolve().as_posix()
        records.append(f"{label}\t{relative}")
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def accumulate_class_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized means of normalized embeddings and class counts."""
    if embeddings.ndim != 2 or labels.ndim != 1:
        raise ValueError("embeddings must be [N,D] and labels must be [N]")
    if embeddings.shape[0] != labels.numel() or embeddings.shape[0] == 0:
        raise ValueError("embeddings and labels must contain the same non-zero N")
    if labels.min().item() < 0 or labels.max().item() >= num_classes:
        raise ValueError("label outside configured class range")
    normalized = F.normalize(embeddings.float(), dim=1, eps=1e-12)
    sums = normalized.new_zeros((num_classes, normalized.shape[1]))
    sums.index_add_(0, labels.long(), normalized)
    counts = torch.bincount(labels.long(), minlength=num_classes)
    missing = torch.where(counts == 0)[0]
    if missing.numel():
        raise ValueError(f"prototype training set is missing classes: {missing.tolist()}")
    prototypes = F.normalize(sums / counts[:, None], dim=1, eps=1e-12)
    return prototypes, counts


def prototype_logits(
    embeddings: torch.Tensor, prototypes: torch.Tensor, *, scale: float
) -> torch.Tensor:
    if embeddings.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("embeddings and prototypes must both be matrices")
    if embeddings.shape[1] != prototypes.shape[1]:
        raise ValueError("embedding and prototype dimensions do not match")
    if scale <= 0:
        raise ValueError("prototype logit scale must be positive")
    return F.linear(
        F.normalize(embeddings.float(), dim=1, eps=1e-12),
        F.normalize(prototypes.float(), dim=1, eps=1e-12),
    ) * float(scale)


@torch.no_grad()
def collect_embeddings(model, loader, device):
    embedding_batches, label_batches = [], []
    for images, labels in loader:
        embedding_batches.append(model.forward_features(images.to(device)).float().cpu())
        label_batches.append(labels.long().cpu())
    if not embedding_batches:
        raise ValueError("empty dataloader")
    return torch.cat(embedding_batches), torch.cat(label_batches)


def metric_summary(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int]:
    predictions = logits.argmax(1)
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    margins = true_logits - masked.max(1).values
    correct = int((predictions == labels).sum())
    return {
        "correct": correct,
        "samples": int(labels.numel()),
        "errors": int(labels.numel()) - correct,
        "accuracy": correct / labels.numel(),
        "ordinary_ce_loss": float(F.cross_entropy(logits, labels)),
        "mean_true_class_margin": float(margins.mean()),
        "p05_true_class_margin": float(torch.quantile(margins, 0.05)),
    }


def selection_key(summary: dict[str, float | int]) -> tuple[int, float, float]:
    return (
        int(summary["errors"]),
        float(summary["ordinary_ce_loss"]),
        -float(summary["mean_true_class_margin"]),
    )


def write_predictions(
    output_path: Path,
    samples: list[tuple[Path, int]],
    labels: torch.Tensor,
    learned_logits: torch.Tensor,
    proto_logits: torch.Tensor,
) -> None:
    learned_probs = learned_logits.softmax(1)
    proto_probs = proto_logits.softmax(1)
    learned_conf, learned_pred = learned_probs.max(1)
    proto_conf, proto_pred = proto_probs.max(1)
    rows = []
    for index, ((path, identity), label) in enumerate(zip(samples, labels.tolist())):
        rows.append({
            "sample_id": str(path),
            "identity": identity,
            "true_class": label,
            "learned_prediction": int(learned_pred[index]),
            "learned_correct": int(learned_pred[index] == label),
            "learned_confidence": float(learned_conf[index]),
            "prototype_prediction": int(proto_pred[index]),
            "prototype_correct": int(proto_pred[index] == label),
            "prototype_confidence": float(proto_conf[index]),
        })
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--prototype-path", required=True)
    parser.add_argument("--build-prototypes", action="store_true")
    parser.add_argument("--overwrite-prototypes", action="store_true")
    parser.add_argument("--partition", choices=["val", "test"], default="val")
    parser.add_argument("--acknowledge-observed-test", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scale", type=float)
    args = parser.parse_args()

    if args.partition == "test" and not args.acknowledge_observed_test:
        parser.error("test evaluation requires --acknowledge-observed-test")
    if args.partition == "test" and args.build_prototypes:
        parser.error("build prototypes during validation, then reuse the frozen artifact for test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = build_model(args.config, args.checkpoint, device)
    split = load_split(args.split_path)
    label_map = build_label_map(split["subjects"])
    num_classes = len(label_map)
    data_dir = Path(args.data_dir)
    transform = get_transforms(
        "val", 224, input_profile=model_cfg.get("input_profile", "legacy")
    )
    scale = float(args.scale or model_cfg.get("arcface_scale", 64.0))
    prototype_path = Path(args.prototype_path)

    if args.build_prototypes:
        if prototype_path.exists() and not args.overwrite_prototypes:
            raise FileExistsError(
                f"prototype artifact already exists: {prototype_path}; use --overwrite-prototypes"
            )
        train_samples = build_image_list(data_dir, split["train"], label_map)
        train_loader = DataLoader(
            PalmVeinDataset(train_samples, transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        train_embeddings, train_labels = collect_embeddings(model, train_loader, device)
        prototypes, counts = accumulate_class_prototypes(
            train_embeddings, train_labels, num_classes=num_classes
        )
        metadata = {
            "method": "normalized training-class mean prototype classifier",
            "source_partition": "train_only",
            "test_partition_inspected": False,
            "checkpoint_sha256": sha(args.checkpoint),
            "config_sha256": sha(args.config),
            "split_sha256": sha(args.split_path),
            "training_sample_manifest_sha256": sample_manifest_sha(train_samples, data_dir),
            "num_classes": num_classes,
            "embedding_dim": int(prototypes.shape[1]),
            "samples": len(train_samples),
            "minimum_samples_per_class": int(counts.min()),
            "maximum_samples_per_class": int(counts.max()),
            "input_profile": model_cfg.get("input_profile", "legacy"),
            "scale": scale,
        }
        prototype_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"prototypes": prototypes.cpu(), "class_counts": counts.cpu(), "metadata": metadata},
            prototype_path,
        )
    else:
        if not prototype_path.is_file():
            raise FileNotFoundError(f"prototype artifact not found: {prototype_path}")

    artifact = torch.load(prototype_path, map_location="cpu", weights_only=False)
    prototypes = artifact["prototypes"].float()
    metadata = artifact["metadata"]
    required = {
        "checkpoint_sha256": sha(args.checkpoint),
        "config_sha256": sha(args.config),
        "split_sha256": sha(args.split_path),
        "num_classes": num_classes,
        "input_profile": model_cfg.get("input_profile", "legacy"),
    }
    mismatches = {
        key: {"artifact": metadata.get(key), "current": value}
        for key, value in required.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"prototype provenance mismatch: {mismatches}")
    if prototypes.shape[0] != num_classes:
        raise ValueError("prototype artifact class count mismatch")

    eval_samples = build_image_list(data_dir, split[args.partition], label_map)
    eval_loader = DataLoader(
        PalmVeinDataset(eval_samples, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    embeddings, labels = collect_embeddings(model, eval_loader, device)
    with torch.no_grad():
        learned_logits = model.classifier(model.dropout(embeddings.to(device))).float().cpu()
        proto_logits = prototype_logits(embeddings, prototypes, scale=scale).cpu()

    learned = metric_summary(learned_logits, labels)
    prototype = metric_summary(proto_logits, labels)
    decision = "prototype_selected" if selection_key(prototype) < selection_key(learned) else "keep_learned_head"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_predictions(
        output / "predictions.csv", eval_samples, labels, learned_logits, proto_logits
    )
    result = {
        "task": "closed_set_identification",
        "partition": args.partition,
        "test_previously_observed_acknowledged": bool(args.acknowledge_observed_test),
        "checkpoint_selection_occurred_on_test": False,
        "prototype_built_from": metadata["source_partition"],
        "prototype_artifact": str(prototype_path),
        "prototype_artifact_sha256": sha(prototype_path),
        "prototype_metadata": metadata,
        "learned_head": learned,
        "prototype_head": prototype,
        "selection_rule": "errors -> ordinary CE loss -> mean true-class margin",
        "validation_decision": decision if args.partition == "val" else None,
        "test_loader_created": args.partition == "test",
        "reported_metrics": ["accuracy_crr", "correct_total"],
        "excluded_metrics": ["eer", "far", "frr", "biometric_auc"],
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

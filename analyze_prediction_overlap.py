"""
Prediction overlap analysis for NAS C12 vs teacher/baseline models.

This diagnostic script evaluates the same test split with:
  - NAS C12 student
  - MobileNetV3Small baseline
  - EfficientNetV2M teacher
  - ResNet50 teacher

It writes per-sample predictions, C12 error rows, and a compact summary to help
decide whether KD should focus on hard samples, margins, or capacity.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from genotypes import dict_to_genotype
from model_eval import EvalNetwork
from palm_vein_dataset import (
    build_label_map,
    build_image_list,
    create_retrain_dataloaders,
    load_split,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_NAS_DIR = ROOT / "nas_results" / "retrain_hwNAS_L0.05_C12_stemds8_834cls"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "prediction_overlap_C12"


MODEL_SPECS = {
    "mobilenet": {
        "display": "MobileNetV3Small",
        "weights": ROOT / "Teacher" / "training_results" / "MobileNetV3Small" / "best_model.pth",
    },
    "effv2m": {
        "display": "EfficientNetV2M",
        "weights": ROOT / "Teacher" / "training_results" / "EfficientNetV2M" / "best_model.pth",
    },
    "resnet50": {
        "display": "ResNet50",
        "weights": ROOT / "Teacher" / "training_results" / "ResNet50" / "best_model.pth",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump prediction overlap for C12 NAS, MobileNetV3Small, EfficientNetV2M, and ResNet50."
    )
    parser.add_argument("--data_dir", default=str(ROOT / "preprocessed_results"))
    parser.add_argument("--split_path", default=str(ROOT / "split_info.json"))
    parser.add_argument("--nas_config", default=str(DEFAULT_NAS_DIR / "config.json"))
    parser.add_argument("--nas_weights", default=str(DEFAULT_NAS_DIR / "best_model.pth"))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--low_margin_threshold", type=float, default=0.10)
    return parser.parse_args()


def unwrap_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise TypeError(f"Unsupported checkpoint object: {type(obj)!r}")


def load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    return unwrap_state_dict(torch.load(path, map_location="cpu"))


def build_nas_model(config_path: Path, weights_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    with config_path.open() as f:
        cfg = json.load(f)

    genotype = dict_to_genotype(cfg["genotype"])
    reduction_indices = cfg.get("reduction_indices")
    if isinstance(reduction_indices, str):
        reduction_indices = [int(x) for x in reduction_indices.split(",") if x.strip()]

    retrain_cfg = cfg.get("retrain_cfg", {})
    dropout = float(retrain_cfg.get("dropout", cfg.get("dropout", 0.3)))

    model = EvalNetwork(
        genotype=genotype,
        C_init=int(cfg["C_init"]),
        num_cells=int(cfg["num_cells"]),
        num_classes=num_classes,
        auxiliary=False,
        dropout=dropout,
        stem_downsample=int(cfg.get("stem_downsample", 2)),
        reduction_indices=reduction_indices,
    )

    state_dict = load_state_dict_file(weights_path)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_auxiliary_head")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[WARN] NAS unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if missing:
        print(f"[WARN] NAS missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    model.to(device)
    model.eval()
    return model


def build_teacher_fallback(model_name: str, num_classes: int) -> nn.Module:
    """Build teacher architecture without downloading ImageNet weights."""
    if model_name == "MobileNetV3Small":
        model = models.mobilenet_v3_small(weights=None)
        in_feat = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=0.5)
        model.classifier[3] = nn.Linear(in_feat, num_classes)
        return model

    if model_name == "EfficientNetV2M":
        model = models.efficientnet_v2_m(weights=None)
        in_feat = model.classifier[1].in_features
        model.classifier[0] = nn.Dropout(p=0.5)
        model.classifier[1] = nn.Linear(in_feat, num_classes)
        return model

    if model_name == "ResNet50":
        model = models.resnet50(weights=None)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_feat, num_classes))
        return model

    raise ValueError(f"Unsupported teacher model: {model_name}")


def build_teacher_model(model_name: str, weights_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    try:
        from Teacher.model_factory import create_model

        model, _ = create_model(model_name, num_classes)
    except Exception as exc:
        print(f"[WARN] Teacher/model_factory failed for {model_name}: {exc}")
        print(f"[WARN] Falling back to torchvision weights=None builder for {model_name}.")
        model = build_teacher_fallback(model_name, num_classes)

    state_dict = load_state_dict_file(weights_path)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def build_test_metadata(data_dir: Path, split_path: Path) -> list[dict[str, Any]]:
    split = load_split(split_path)
    label_map = build_label_map(split["subjects"])
    samples = build_image_list(data_dir, split["test"], label_map)

    metadata: list[dict[str, Any]] = []
    for idx, (path, label) in enumerate(samples):
        path = Path(path)
        metadata.append(
            {
                "index": idx,
                "subject_id": path.parent.name,
                "filename": path.name,
                "path": str(path),
                "label": int(label),
            }
        )
    return metadata


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
    topk: int = 5,
) -> dict[str, list[Any]]:
    all_topk_labels: list[list[int]] = []
    all_topk_probs: list[list[float]] = []
    all_true_probs: list[float] = []
    all_true_ranks: list[int | None] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)
        k = min(topk, probs.shape[1])
        top_probs, top_labels = probs.topk(k, dim=1)

        sorted_labels = probs.argsort(dim=1, descending=True)
        for row_idx in range(labels.size(0)):
            true_label = int(labels[row_idx].item())
            rank_positions = (sorted_labels[row_idx] == true_label).nonzero(as_tuple=False)
            rank = int(rank_positions[0].item()) + 1 if rank_positions.numel() else None
            all_true_ranks.append(rank)
            all_true_probs.append(float(probs[row_idx, true_label].item()))

        all_labels.extend(int(x) for x in labels.cpu().tolist())
        all_topk_labels.extend([[int(x) for x in row] for row in top_labels.cpu().tolist()])
        all_topk_probs.extend([[float(x) for x in row] for row in top_probs.cpu().tolist()])

    return {
        "labels": all_labels,
        "topk_labels": all_topk_labels,
        "topk_probs": all_topk_probs,
        "true_probs": all_true_probs,
        "true_ranks": all_true_ranks,
    }


def row_diagnosis(row: dict[str, Any], low_margin_threshold: float) -> str:
    teacher_all_correct = bool(row["effv2m_correct"] and row["resnet50_correct"])
    all_models_wrong = not bool(row["mobilenet_correct"] or row["effv2m_correct"] or row["resnet50_correct"])
    low_margin = float(row["c12_margin_top1_top2"]) < low_margin_threshold
    true_in_top5 = bool(row["c12_true_in_top5"])

    tags = []
    if teacher_all_correct:
        tags.append("teacher_all_correct")
    if row["mobilenet_correct"] and not teacher_all_correct:
        tags.append("mobilenet_correct_only")
    if all_models_wrong:
        tags.append("all_models_wrong")
    if low_margin:
        tags.append("low_margin")
    if true_in_top5:
        tags.append("true_in_top5")
    return ";".join(tags) if tags else "c12_specific_error"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_recommendations(summary: dict[str, Any]) -> list[str]:
    c12_errors = summary["c12_error_count"]
    if c12_errors == 0:
        return ["C12 already solves all test samples; no KD diagnostic needed."]

    recs = []
    if summary["c12_errors_teacher_all_correct"] == c12_errors:
        recs.append("Teachers solve all C12 errors; prioritize hard-sample KD, margin KD, or top-k distillation.")
    if summary["c12_errors_mobilenet_correct"] > 0:
        recs.append("MobileNet solves some C12 errors; compare MobileNet/C12 features or try multi-teacher KD.")
    if summary["c12_errors_true_in_top5"] > 0:
        recs.append("Some true labels are in C12 top-5; margin-ranking or ArcFace/SupCon fine-tuning is promising.")
    if summary["c12_errors_true_in_top5"] == 0:
        recs.append("True labels are outside C12 top-5; inspect image quality and consider C14/C16 capacity.")
    if summary["c12_errors_all_models_wrong"] > 0:
        recs.append("Some samples are wrong for all models; audit labels/split/image quality before more KD.")
    return recs


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Prediction Overlap Summary",
        "",
        "## Accuracy",
    ]
    for name, acc in summary["accuracy"].items():
        lines.append(f"- {name}: {acc * 100:.2f}%")
    lines.extend(
        [
            "",
            "## C12 Error Analysis",
            f"- C12 error count: {summary['c12_error_count']}",
            f"- C12 errors where EfficientNetV2M and ResNet50 are correct: {summary['c12_errors_teacher_all_correct']}",
            f"- C12 errors where MobileNetV3Small is correct: {summary['c12_errors_mobilenet_correct']}",
            f"- C12 errors where true label is in C12 top-5: {summary['c12_errors_true_in_top5']}",
            f"- C12 errors with low margin: {summary['c12_errors_low_margin']}",
            f"- C12 errors wrong for all compared models: {summary['c12_errors_all_models_wrong']}",
            "",
            "## Recommendations",
        ]
    )
    lines.extend(f"- {rec}" for rec in summary["recommendations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    split_path = Path(args.split_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader, info = create_retrain_dataloaders(
        data_dir=data_dir,
        split_path=split_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=False,
    )
    num_classes = int(info["num_classes"])
    metadata = build_test_metadata(data_dir, split_path)
    if len(metadata) != len(test_loader.dataset):
        raise RuntimeError(
            f"Metadata/test dataset length mismatch: {len(metadata)} vs {len(test_loader.dataset)}"
        )

    print("Loading C12 NAS...")
    models_to_eval: dict[str, nn.Module] = {
        "c12": build_nas_model(Path(args.nas_config), Path(args.nas_weights), num_classes, device)
    }

    for key, spec in MODEL_SPECS.items():
        print(f"Loading {spec['display']}...")
        models_to_eval[key] = build_teacher_model(spec["display"], spec["weights"], num_classes, device)

    predictions = {}
    for key, model in models_to_eval.items():
        print(f"Evaluating {key}...")
        predictions[key] = collect_predictions(model, test_loader, device, topk=5)

    labels = predictions["c12"]["labels"]
    rows: list[dict[str, Any]] = []
    for idx, meta in enumerate(metadata):
        true_label = int(labels[idx])
        c12_top_labels = predictions["c12"]["topk_labels"][idx]
        c12_top_probs = predictions["c12"]["topk_probs"][idx]

        row: dict[str, Any] = {
            **meta,
            "true_label": true_label,
            "c12_top1": c12_top_labels[0],
            "c12_top1_prob": c12_top_probs[0],
            "c12_top2": c12_top_labels[1] if len(c12_top_labels) > 1 else None,
            "c12_top2_prob": c12_top_probs[1] if len(c12_top_probs) > 1 else None,
            "c12_margin_top1_top2": c12_top_probs[0] - (c12_top_probs[1] if len(c12_top_probs) > 1 else 0.0),
            "c12_top5_labels": "|".join(str(x) for x in c12_top_labels),
            "c12_top5_probs": "|".join(f"{x:.8f}" for x in c12_top_probs),
            "c12_true_rank": predictions["c12"]["true_ranks"][idx],
            "c12_true_prob": predictions["c12"]["true_probs"][idx],
            "c12_true_in_top5": true_label in c12_top_labels,
        }

        for key in ("c12", "mobilenet", "effv2m", "resnet50"):
            pred = int(predictions[key]["topk_labels"][idx][0])
            prob = float(predictions[key]["topk_probs"][idx][0])
            row[f"{key}_pred"] = pred
            row[f"{key}_conf"] = prob
            row[f"{key}_correct"] = pred == true_label

        row["diagnosis"] = row_diagnosis(row, args.low_margin_threshold) if not row["c12_correct"] else "c12_correct"
        rows.append(row)

    c12_errors = [row for row in rows if not row["c12_correct"]]

    accuracy = {
        key: sum(1 for row in rows if row[f"{key}_correct"]) / len(rows)
        for key in ("c12", "mobilenet", "effv2m", "resnet50")
    }
    summary = {
        "num_test_samples": len(rows),
        "accuracy": accuracy,
        "c12_error_count": len(c12_errors),
        "c12_errors_teacher_all_correct": sum(
            1 for row in c12_errors if row["effv2m_correct"] and row["resnet50_correct"]
        ),
        "c12_errors_mobilenet_correct": sum(1 for row in c12_errors if row["mobilenet_correct"]),
        "c12_errors_true_in_top5": sum(1 for row in c12_errors if row["c12_true_in_top5"]),
        "c12_errors_low_margin": sum(
            1 for row in c12_errors if float(row["c12_margin_top1_top2"]) < args.low_margin_threshold
        ),
        "c12_errors_all_models_wrong": sum(
            1
            for row in c12_errors
            if not (row["mobilenet_correct"] or row["effv2m_correct"] or row["resnet50_correct"])
        ),
        "overlap_error_counts": {
            "c12_and_mobilenet": sum(1 for row in rows if not row["c12_correct"] and not row["mobilenet_correct"]),
            "c12_and_effv2m": sum(1 for row in rows if not row["c12_correct"] and not row["effv2m_correct"]),
            "c12_and_resnet50": sum(1 for row in rows if not row["c12_correct"] and not row["resnet50_correct"]),
        },
        "low_margin_threshold": args.low_margin_threshold,
    }
    summary["recommendations"] = make_recommendations(summary)

    write_csv(output_dir / "predictions_all.csv", rows)
    write_csv(output_dir / "c12_errors.csv", c12_errors)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_summary_md(output_dir / "summary.md", summary)

    print("\nAccuracy:")
    for key, acc in accuracy.items():
        print(f"  {key:10s}: {acc * 100:.2f}%")
    print(f"\nC12 errors: {len(c12_errors)}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

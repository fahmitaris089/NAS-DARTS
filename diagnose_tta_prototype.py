"""Diagnostic TTA and prototype evaluation for NAS KD checkpoints.

This script does not train or save model checkpoints. It evaluates whether the
remaining errors can be corrected by deterministic test-time augmentation (TTA)
or by nearest-prototype / classifier-prototype hybrid scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from genotypes import dict_to_genotype
from model_eval import EvalNetwork
from nas_config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE
from palm_vein_dataset import build_image_list, build_label_map, load_split


FOCUS_FILENAMES = {"277_6.bmp", "504_4.bmp"}


@dataclass(frozen=True)
class TTAView:
    name: str
    rotate: float = 0.0
    translate: tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0


DEFAULT_TTA_VIEWS = [
    "original",
    "rot3",
    "rot-3",
    "shiftx3",
    "shiftx-3",
    "shifty3",
    "shifty-3",
    "scale102",
    "scale98",
]


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("L")
        image = self.transform(image)
        return image, int(label), idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose TTA/prototype correction for a NAS checkpoint")
    parser.add_argument("--student_config", required=True)
    parser.add_argument("--student_weights", required=True)
    parser.add_argument("--data_dir", default=str(ROOT / "preprocessed_results"))
    parser.add_argument("--split_path", default=str(ROOT / "split_info.json"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["tta", "prototype", "hybrid", "all"], default="all")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--input_size", type=int, default=INPUT_SIZE)
    parser.add_argument("--tta_views", default=",".join(DEFAULT_TTA_VIEWS))
    parser.add_argument("--prototype_weights", default="0.1,0.2,0.3,0.5,1.0")
    parser.add_argument("--logit_temperature", type=float, default=1.0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise TypeError(f"Unsupported checkpoint object: {type(obj)!r}")


def parse_reduction_indices(raw_value) -> list[int] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return [int(x) for x in raw_value]
    if isinstance(raw_value, str):
        return [int(x.strip()) for x in raw_value.split(",") if x.strip()]
    raise TypeError(f"Unsupported reduction_indices type: {type(raw_value)}")


def build_model(config_path: Path, weights_path: Path, num_classes: int, device: torch.device) -> EvalNetwork:
    cfg = load_json(config_path)
    genotype = dict_to_genotype(cfg["genotype"])
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
        reduction_indices=parse_reduction_indices(cfg.get("reduction_indices")),
    )
    state_dict = unwrap_state_dict(torch.load(weights_path, map_location="cpu"))
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_auxiliary_head")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    model.to(device)
    model.eval()
    return model


class GrayscaleToRGB:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.repeat(3, 1, 1) if tensor.shape[0] == 1 else tensor


def base_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        GrayscaleToRGB(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def parse_tta_view(name: str) -> TTAView:
    name = name.strip()
    if name == "original":
        return TTAView(name="original")
    if name.startswith("rot"):
        return TTAView(name=name, rotate=float(name.replace("rot", "")))
    if name.startswith("scale"):
        raw = float(name.replace("scale", ""))
        return TTAView(name=name, scale=raw / 100.0)
    if name.startswith("shiftx"):
        raw = float(name.replace("shiftx", "")) / 100.0
        return TTAView(name=name, translate=(raw, 0.0))
    if name.startswith("shifty"):
        raw = float(name.replace("shifty", "")) / 100.0
        return TTAView(name=name, translate=(0.0, raw))
    raise ValueError(f"Unsupported TTA view: {name}")


def tta_transform(view: TTAView, input_size: int):
    ops = [transforms.Resize((input_size, input_size))]
    if view.name != "original":
        translate_px = (int(round(abs(view.translate[0]) * input_size)), int(round(abs(view.translate[1]) * input_size)))
        # torchvision affine translation is in pixels. Sign is carried by the tuple values below.
        tx = int(round(view.translate[0] * input_size))
        ty = int(round(view.translate[1] * input_size))
        ops.append(
            transforms.Lambda(
                lambda img, angle=view.rotate, translate=(tx, ty), scale=view.scale: transforms.functional.affine(
                    img,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=[0.0, 0.0],
                    fill=0,
                )
            )
        )
    ops.extend([
        transforms.ToTensor(),
        GrayscaleToRGB(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(ops)


def load_split_samples(data_dir: Path, split_path: Path):
    split = load_split(split_path)
    label_map = build_label_map(split["subjects"])
    train_samples = build_image_list(data_dir, split["train"], label_map)
    test_samples = build_image_list(data_dir, split["test"], label_map)
    train_samples = [(Path(path), int(label)) for path, label in train_samples]
    test_samples = [(Path(path), int(label)) for path, label in test_samples]
    return train_samples, test_samples, len(label_map)


def sample_metadata(samples: list[tuple[Path, int]]) -> list[dict[str, Any]]:
    rows = []
    for idx, (path, label) in enumerate(samples):
        rows.append({
            "index": idx,
            "subject_id": path.parent.name,
            "filename": path.name,
            "path": str(path),
            "label": int(label),
        })
    return rows


@torch.no_grad()
def collect_logits_and_embeddings(model: EvalNetwork, loader, device: torch.device, embeddings: bool = False):
    all_logits = []
    all_embeddings = []
    all_labels = []
    all_indices = []
    model.eval()
    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        if embeddings:
            logits, emb = model.forward_with_embeddings(images)
            all_embeddings.append(emb.detach().cpu())
        else:
            logits = model(images)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_indices.append(indices.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    indices = torch.cat(all_indices, dim=0)
    order = torch.argsort(indices)
    if embeddings:
        emb = torch.cat(all_embeddings, dim=0)
        return logits[order], labels[order], emb[order]
    return logits[order], labels[order]


def accuracy_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, int, int]:
    pred = scores.argmax(dim=1)
    correct = int(pred.eq(labels).sum().item())
    total = int(labels.numel())
    return correct / total if total else 0.0, correct, total


def compute_prototypes(train_embeddings: torch.Tensor, train_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    emb = F.normalize(train_embeddings, dim=1)
    prototypes = torch.zeros(num_classes, emb.size(1), dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.float32)
    prototypes.index_add_(0, train_labels, emb)
    counts.index_add_(0, train_labels, torch.ones_like(train_labels, dtype=torch.float32))
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).view(-1).tolist()
        raise RuntimeError(f"Missing train samples for classes: {missing[:10]}")
    prototypes = prototypes / counts.unsqueeze(1).clamp_min(1.0)
    return F.normalize(prototypes, dim=1)


def normalized_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = logits / max(float(temperature), 1e-6)
    mean = logits.mean(dim=1, keepdim=True)
    std = logits.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (logits - mean) / std


def rows_from_scores(
    metadata: list[dict[str, Any]],
    labels: torch.Tensor,
    method_scores: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows = []
    baseline_pred = method_scores["classifier"].argmax(dim=1)
    for i, meta in enumerate(metadata):
        true_label = int(labels[i].item())
        row = {**meta, "true_label": true_label}
        for name, scores in method_scores.items():
            probs = torch.softmax(scores[i], dim=0)
            pred = int(scores[i].argmax().item())
            row[f"{name}_pred"] = pred
            row[f"{name}_conf"] = float(probs[pred].item())
            row[f"{name}_correct"] = pred == true_label
        row["baseline_correct"] = int(baseline_pred[i].item()) == true_label
        row["focus_error"] = meta["filename"] in FOCUS_FILENAMES or not row["baseline_correct"]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_method(name: str, scores: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    acc, correct, total = accuracy_from_scores(scores, labels)
    return {"method": name, "accuracy": acc, "correct": correct, "total": total}


def build_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# TTA / Prototype Diagnostic Summary", ""]
    lines.append("## Results")
    for item in summary["results"]:
        lines.append(f"- {item['method']}: {item['accuracy'] * 100:.2f}% ({item['correct']}/{item['total']})")
    lines.extend(["", "## Best", f"- {summary['best_method']}: {summary['best_accuracy'] * 100:.2f}%", ""])
    lines.append("## Focus Errors")
    for item in summary["focus_errors"]:
        lines.append(
            f"- {item['filename']}: true={item['true_label']} "
            f"classifier={item.get('classifier_pred')} "
            f"best={item.get(summary['best_method'] + '_pred', 'n/a')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    split_path = Path(args.split_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_samples, test_samples, num_classes = load_split_samples(data_dir, split_path)
    metadata = sample_metadata(test_samples)
    model = build_model(Path(args.student_config), Path(args.student_weights), num_classes, device)

    base_test_loader = DataLoader(
        ImagePathDataset(test_samples, base_transform(args.input_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    classifier_logits, labels, test_embeddings = collect_logits_and_embeddings(
        model, base_test_loader, device, embeddings=True
    )

    method_scores: dict[str, torch.Tensor] = {"classifier": classifier_logits}
    results = [summarize_method("classifier", classifier_logits, labels)]

    if args.mode in {"tta", "all"}:
        view_names = [x.strip() for x in args.tta_views.split(",") if x.strip()]
        tta_sum = torch.zeros_like(classifier_logits)
        tta_rows_extra = {}
        for view_name in view_names:
            view = parse_tta_view(view_name)
            loader = DataLoader(
                ImagePathDataset(test_samples, tta_transform(view, args.input_size)),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            logits, _labels = collect_logits_and_embeddings(model, loader, device, embeddings=False)
            tta_sum += logits
            method_scores[f"tta_{view_name}"] = logits
            results.append(summarize_method(f"tta_{view_name}", logits, labels))
        tta_avg = tta_sum / max(len(view_names), 1)
        method_scores["tta_avg"] = tta_avg
        results.append(summarize_method("tta_avg", tta_avg, labels))

    if args.mode in {"prototype", "hybrid", "all"}:
        train_loader = DataLoader(
            ImagePathDataset(train_samples, base_transform(args.input_size)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        _train_logits, train_labels, train_embeddings = collect_logits_and_embeddings(
            model, train_loader, device, embeddings=True
        )
        prototypes = compute_prototypes(train_embeddings, train_labels, num_classes)
        proto_scores = F.normalize(test_embeddings, dim=1) @ prototypes.t()
        method_scores["prototype"] = proto_scores
        results.append(summarize_method("prototype", proto_scores, labels))

        if args.mode in {"hybrid", "all"}:
            logit_scores = normalized_logits(classifier_logits, args.logit_temperature)
            weights = [float(x.strip()) for x in args.prototype_weights.split(",") if x.strip()]
            for weight in weights:
                name = f"hybrid_w{weight:g}"
                hybrid_scores = logit_scores + weight * proto_scores
                method_scores[name] = hybrid_scores
                results.append(summarize_method(name, hybrid_scores, labels))

    rows = rows_from_scores(metadata, labels, method_scores)
    focus_rows = [row for row in rows if row["focus_error"]]

    best = max(results, key=lambda item: (item["accuracy"], item["correct"]))
    summary = {
        "student_config": str(args.student_config),
        "student_weights": str(args.student_weights),
        "mode": args.mode,
        "num_test_samples": int(labels.numel()),
        "results": results,
        "best_method": best["method"],
        "best_accuracy": best["accuracy"],
        "best_correct": best["correct"],
        "focus_errors": focus_rows,
        "tta_views": args.tta_views,
        "prototype_weights": args.prototype_weights,
    }

    if any(name.startswith("tta_") for name in method_scores):
        tta_rows = rows_from_scores(
            metadata,
            labels,
            {k: v for k, v in method_scores.items() if k == "classifier" or k.startswith("tta_")},
        )
        write_csv(output_dir / "predictions_tta.csv", tta_rows)
    if "prototype" in method_scores:
        proto_rows = rows_from_scores(
            metadata,
            labels,
            {k: v for k, v in method_scores.items() if k == "classifier" or k == "prototype" or k.startswith("hybrid_")},
        )
        write_csv(output_dir / "predictions_prototype.csv", proto_rows)

    write_csv(output_dir / "predictions_all.csv", rows)
    write_csv(output_dir / "error_focus.csv", focus_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_summary_md(output_dir / "summary.md", summary)

    print("\nResults:")
    for item in results:
        print(f"  {item['method']:18s}: {item['accuracy'] * 100:.2f}% ({item['correct']}/{item['total']})")
    print(f"\nBest: {best['method']} = {best['accuracy'] * 100:.2f}% ({best['correct']}/{best['total']})")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

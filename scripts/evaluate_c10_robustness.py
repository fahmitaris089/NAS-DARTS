#!/usr/bin/env python3
"""Validation-only deterministic robustness suite for frozen C10 systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF

ROOT = Path(__file__).resolve().parents[1]
NAS = ROOT / "Eksperimen_Hardware_Aware_PDARTS" / "src" / "nas"
sys.path[:0] = [str(ROOT), str(NAS)]

from palm_input_preprocessing import ApplyInputProfile  # noqa: E402
from palm_vein_dataset import GrayscaleToRGB, build_label_map, load_split  # noqa: E402
from scripts.evaluate_frozen_identification import build_model  # noqa: E402


VARIANTS: tuple[tuple[str, str, float], ...] = (
    ("clean", "clean", 0.0),
    ("rotation_m15", "rotation", -15.0),
    ("rotation_p15", "rotation", 15.0),
    ("translate_x_m08", "translate_x", -0.08),
    ("translate_x_p08", "translate_x", 0.08),
    ("translate_y_m08", "translate_y", -0.08),
    ("translate_y_p08", "translate_y", 0.08),
    ("scale_090", "scale", 0.90),
    ("scale_112", "scale", 1.12),
    ("gamma_065", "gamma", 0.65),
    ("gamma_145", "gamma", 1.45),
    ("contrast_065", "contrast", 0.65),
    ("contrast_135", "contrast", 1.35),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ValidationVariantDataset(Dataset):
    def __init__(self, samples, variant, input_profile, input_size=224):
        self.samples = samples
        self.variant = variant
        self.input_profile = input_profile
        self.input_size = int(input_size)
        self.tail = transforms.Compose([
            ApplyInputProfile(input_profile),
            transforms.ToTensor(),
            GrayscaleToRGB(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def _corrupt(self, image):
        _, operation, value = self.variant
        image = image.convert("L").resize(
            (self.input_size, self.input_size), Image.Resampling.BILINEAR,
        )
        if operation == "clean":
            return image
        if operation == "rotation":
            return TF.rotate(
                image, value, interpolation=InterpolationMode.BILINEAR, fill=0,
            )
        if operation in {"translate_x", "translate_y"}:
            pixels = int(round(value * self.input_size))
            translate = [pixels, 0] if operation == "translate_x" else [0, pixels]
            return TF.affine(
                image, angle=0.0, translate=translate, scale=1.0, shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR, fill=0,
            )
        if operation == "scale":
            return TF.affine(
                image, angle=0.0, translate=[0, 0], scale=value, shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR, fill=0,
            )
        if operation == "gamma":
            return TF.adjust_gamma(image, value)
        if operation == "contrast":
            return TF.adjust_contrast(image, value)
        raise ValueError(f"Unsupported corruption operation: {operation}")

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.tail(self._corrupt(image))
        return tensor, int(label)


@torch.inference_mode()
def evaluate_variant(model, loader, device):
    correct = 0
    total = 0
    total_loss = 0.0
    margin_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        predictions = logits.argmax(1)
        correct += predictions.eq(labels).sum().item()
        true_logits = logits.gather(1, labels[:, None]).squeeze(1)
        competing = logits.clone()
        competing.scatter_(1, labels[:, None], float("-inf"))
        margin_sum += (true_logits - competing.max(1).values).sum().item()
        total += labels.numel()
    return {
        "correct": int(correct),
        "samples": int(total),
        "errors": int(total - correct),
        "accuracy": float(correct / total),
        "ordinary_ce_loss": float(total_loss / total),
        "mean_true_class_margin": float(margin_sum / total),
    }


def checkpoint_candidates(run_dir: Path, explicit: list[Path]) -> list[Path]:
    if explicit:
        paths = explicit
    else:
        names = (
            "best_screening.pth", "best_by_val_loss.pth", "best_model.pth",
            "best_by_val_acc.pth", "best_val_acc_model.pth", "last_model.pth",
        )
        paths = [run_dir / name for name in names if (run_dir / name).exists()]
    unique = []
    hashes = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest not in hashes:
            unique.append(path)
            hashes.add(digest)
    if not unique:
        raise FileNotFoundError(f"No checkpoint candidates found in {run_dir}")
    return unique


def load_validation_samples(data_dir: Path, split: dict):
    label_map = build_label_map(split["subjects"])
    samples = []
    for subject, filename in split["val"]:
        path = data_dir / str(subject) / str(filename)
        if not path.exists():
            raise FileNotFoundError(path)
        samples.append((path, label_map[str(subject)]))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--split-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Diagnostic-only subset; selection rejects incomplete suites",
    )
    args = parser.parse_args()

    config = args.config or (args.run_dir / "config.json")
    for path in (config, args.split_path):
        if not path.exists():
            raise FileNotFoundError(path)
    split = load_split(args.split_path)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    initial_path = config_payload.get("initial_weights")
    initial_sha256 = None
    if initial_path:
        resolved_initial = Path(str(initial_path).replace("\\", "/"))
        if not resolved_initial.is_absolute():
            resolved_initial = ROOT / resolved_initial
        if not resolved_initial.exists():
            raise FileNotFoundError(resolved_initial)
        initial_sha256 = sha256_file(resolved_initial)
    samples = load_validation_samples(args.data_dir, split)
    declared_validation_samples = len(samples)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    candidates = checkpoint_candidates(args.run_dir, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []
    for checkpoint in candidates:
        model, cfg = build_model(config, checkpoint, device)
        input_profile = str(cfg.get("input_profile", "legacy"))
        variants = {}
        for variant in VARIANTS:
            dataset = ValidationVariantDataset(samples, variant, input_profile)
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            )
            variants[variant[0]] = evaluate_variant(model, loader, device)
        clean = variants["clean"]
        corrupted = [metrics for name, metrics in variants.items() if name != "clean"]
        summary = {
            "clean_validation_errors": clean["errors"],
            "worst_case_corruption_errors": max(item["errors"] for item in corrupted),
            "total_corruption_errors": sum(item["errors"] for item in corrupted),
            "mean_corruption_ce": sum(item["ordinary_ce_loss"] for item in corrupted) / len(corrupted),
            "clean_ce": clean["ordinary_ce_loss"],
            "clean_true_class_margin": clean["mean_true_class_margin"],
        }
        summary["selection_key"] = [
            summary["clean_validation_errors"],
            summary["worst_case_corruption_errors"],
            summary["total_corruption_errors"],
            summary["mean_corruption_ce"],
            summary["clean_ce"],
            -summary["clean_true_class_margin"],
        ]
        all_results.append({
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "input_profile": input_profile,
            "stem_pool": cfg.get("stem_pool", "max"),
            "consistency_mode": cfg.get("consistency_mode", "none"),
            "summary": summary,
            "variants": variants,
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    winner = min(all_results, key=lambda item: tuple(item["summary"]["selection_key"]))
    payload = {
        "partition": "validation_only",
        "test_loader_created": False,
        "test_partition_inspected": False,
        "selection_rule": (
            "clean errors -> worst corruption errors -> total corruption errors -> "
            "mean corruption CE -> clean CE -> clean margin"
        ),
        "run_dir": str(args.run_dir),
        "config": str(config),
        "config_sha256": sha256_file(config),
        "initial_student_sha256": initial_sha256,
        "split_path": str(args.split_path),
        "split_sha256": sha256_file(args.split_path),
        "validation_samples": len(samples),
        "declared_validation_samples": declared_validation_samples,
        "suite_complete": len(samples) == declared_validation_samples,
        "variant_names": [variant[0] for variant in VARIANTS],
        "winner": winner,
        "checkpoint_candidates": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

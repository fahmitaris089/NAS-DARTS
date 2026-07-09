"""Average NAS/KD checkpoints and evaluate the averaged model.

This utility is intended for SWA/checkpoint averaging experiments after a KD
run. It writes a normal KD-like output directory containing best_model.pth,
last_model.pth, config.json, and test_results.json so the existing ONNX export
and prediction-overlap tools can consume it directly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from genotypes import dict_to_genotype
from model_eval import EvalNetwork
from palm_vein_dataset import create_retrain_dataloaders


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_path_text(path_value: str | Path) -> str:
    return str(path_value).replace("\\", os.sep)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(normalize_path_text(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_reduction_indices(raw_value) -> list[int] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return [int(x) for x in raw_value]
    if isinstance(raw_value, str):
        return [int(x.strip()) for x in raw_value.split(",") if x.strip()]
    raise TypeError(f"Unsupported reduction_indices type: {type(raw_value)}")


def extract_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise TypeError("Checkpoint must be a state_dict or contain a state_dict-like field")


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    return {k: v for k, v in state_dict.items() if not k.startswith("_auxiliary_head")}


def average_state_dicts(checkpoint_paths: list[Path]) -> dict[str, torch.Tensor]:
    if len(checkpoint_paths) < 2:
        raise ValueError("Need at least two checkpoints to average.")

    state_dicts = [load_state_dict(path) for path in checkpoint_paths]
    reference_keys = set(state_dicts[0].keys())
    for path, state_dict in zip(checkpoint_paths[1:], state_dicts[1:]):
        if set(state_dict.keys()) != reference_keys:
            missing = sorted(reference_keys - set(state_dict.keys()))[:10]
            extra = sorted(set(state_dict.keys()) - reference_keys)[:10]
            raise RuntimeError(
                f"Checkpoint keys do not match: {path}\n"
                f"missing={missing}\nextra={extra}"
            )

    averaged = {}
    first = state_dicts[0]
    last = state_dicts[-1]
    for key, first_tensor in first.items():
        if not torch.is_tensor(first_tensor):
            continue

        tensors = [state_dict[key] for state_dict in state_dicts]
        if any(t.shape != first_tensor.shape for t in tensors):
            shapes = [tuple(t.shape) for t in tensors]
            raise RuntimeError(f"Shape mismatch for {key}: {shapes}")

        if torch.is_floating_point(first_tensor):
            acc = torch.zeros_like(first_tensor, dtype=torch.float32)
            for tensor in tensors:
                acc.add_(tensor.detach().to(dtype=torch.float32))
            avg = acc.div_(len(tensors)).to(dtype=first_tensor.dtype)
            averaged[key] = avg
        else:
            # e.g. BatchNorm num_batches_tracked. Copy from last checkpoint.
            averaged[key] = last[key].detach().clone()

    return averaged


def build_model(config_path: Path, state_dict: dict[str, torch.Tensor], num_classes: int, device: torch.device) -> EvalNetwork:
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
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    return model.to(device)


@torch.no_grad()
def evaluate(model: EvalNetwork, loader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        pred = logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.numel()
        total_loss += loss.item() * labels.numel()

    return {
        "acc": correct / total if total else 0.0,
        "loss": total_loss / total if total else 0.0,
        "correct": int(correct),
        "total": int(total),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average NAS/KD checkpoints and evaluate the result")
    parser.add_argument("--student_config", required=True)
    parser.add_argument("--checkpoint_paths", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default="preprocessed_results")
    parser.add_argument("--split_path", default="split_info.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    student_config = resolve_path(args.student_config)
    checkpoint_paths = [resolve_path(path) for path in args.checkpoint_paths]

    for path in [student_config, *checkpoint_paths]:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("Checkpoint Averaging / SWA")
    print("=" * 70)
    print(f"Device       : {device}")
    print(f"Student cfg  : {student_config}")
    print("Checkpoints  :")
    for path in checkpoint_paths:
        print(f"  - {path}")
    print(f"Output       : {output_dir}")

    _, val_loader, test_loader, info = create_retrain_dataloaders(
        data_dir=str(resolve_path(args.data_dir)),
        split_path=str(resolve_path(args.split_path)),
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers,
        use_augmentation=False,
    )

    averaged_state = average_state_dicts(checkpoint_paths)
    model = build_model(student_config, averaged_state, info["num_classes"], device)

    val = evaluate(model, val_loader, device)
    test = evaluate(model, test_loader, device)

    torch.save(averaged_state, output_dir / "best_model.pth")
    torch.save(averaged_state, output_dir / "last_model.pth")

    config = {
        "created_at": datetime.now().isoformat(),
        "method": "checkpoint_averaging",
        "student_config_path": str(student_config),
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "num_checkpoints": len(checkpoint_paths),
        "data_dir": str(resolve_path(args.data_dir)),
        "split_path": str(resolve_path(args.split_path)),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "input_size": args.input_size,
        "seed": args.seed,
        "num_classes": info["num_classes"],
        "val": val,
        "test": test,
    }
    save_json(output_dir / "config.json", config)
    save_json(output_dir / "test_results.json", {
        "method": "checkpoint_averaging",
        "test_accuracy": test["acc"],
        "test_correct": test["correct"],
        "test_total": test["total"],
        "test_loss": test["loss"],
        "val_accuracy": val["acc"],
        "val_correct": val["correct"],
        "val_total": val["total"],
        "val_loss": val["loss"],
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
    })

    print("=" * 70)
    print(f"VAL ACC  : {val['acc'] * 100:.2f}% ({val['correct']}/{val['total']}) loss={val['loss']:.4f}")
    print(f"TEST ACC : {test['acc'] * 100:.2f}% ({test['correct']}/{test['total']}) loss={test['loss']:.4f}")
    print(f"Saved    : {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

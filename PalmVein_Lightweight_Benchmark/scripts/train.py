#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import load_json, save_json, select_device, set_seed
from src.data.dataset import build_dataloaders, load_dataset_config, validate_dataset
from src.models import MODEL_NAMES, build_model, count_parameters
from src.models.factory import PRETRAINED_MODELS
from src.training.engine import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train one controlled palm-vein benchmark run")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--protocol", choices=("scratch", "pretrained"), default="scratch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.protocol == "pretrained" and args.model not in PRETRAINED_MODELS:
        raise SystemExit(f"ERROR: {args.model} has no official pretrained weights; result is N/A by protocol.")
    protocol_path = "configs/scratch_600e.json" if args.protocol == "scratch" else "configs/pretrained_200e.json"
    protocol = load_json(protocol_path)
    if args.epochs is not None:
        protocol["epochs"] = args.epochs
    if args.batch_size is not None:
        protocol["batch_size"] = args.batch_size
    if args.num_workers is not None:
        protocol["num_workers"] = args.num_workers
    dataset_config = load_dataset_config()
    validation = validate_dataset(dataset_config, verify_images=True)
    set_seed(args.seed)
    device = select_device(args.device)
    model = build_model(args.model, num_classes=int(dataset_config["expected_classes"]), pretrained=args.protocol == "pretrained")
    loaders, label_map = build_dataloaders(dataset_config, protocol, args.seed)
    run_name = f"{args.model}/seed_{args.seed}"
    result_dir = PROJECT_ROOT / "results" / args.protocol / run_name
    checkpoint_dir = PROJECT_ROOT / "artifacts/checkpoints" / args.protocol / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "protocol": args.protocol,
        "seed": args.seed,
        "num_classes": len(label_map),
        "parameters": count_parameters(model),
        "split_sha256": validation["split_sha256"],
    }
    run_config = {
        **metadata,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dataset": dataset_config,
        "training": protocol,
    }
    save_json(run_config, result_dir / "run_config.json")
    result = run_training(model, loaders, protocol, device, checkpoint_dir, result_dir, metadata, args.resume)
    save_json({**metadata, **result}, result_dir / "test_results.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

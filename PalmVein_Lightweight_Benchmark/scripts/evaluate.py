#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import load_json, save_json, select_device, set_seed
from src.data import build_dataloaders, load_dataset_config
from src.evaluation.metrics import evaluate_classifier
from src.models import build_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved best checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = state.get("metadata", {})
    model_name = metadata.get("model")
    protocol_name = metadata.get("protocol", "scratch")
    seed = int(metadata.get("seed", 42))
    if not model_name:
        raise SystemExit("Checkpoint has no model metadata")
    protocol = load_json("configs/scratch_600e.json" if protocol_name == "scratch" else "configs/pretrained_200e.json")
    dataset_config = load_dataset_config()
    set_seed(seed)
    loaders, _ = build_dataloaders(dataset_config, protocol, seed)
    model = build_model(model_name, int(dataset_config["expected_classes"]), pretrained=protocol_name == "pretrained")
    model.load_state_dict(state["model_state"], strict=True)
    device = select_device(args.device)
    model.to(device)
    metrics = evaluate_classifier(model, loaders[args.split], nn.CrossEntropyLoss(), device)
    payload = {**metadata, "evaluated_split": args.split, **metrics}
    if args.output:
        save_json(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

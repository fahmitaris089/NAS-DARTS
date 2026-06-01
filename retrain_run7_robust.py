#!/usr/bin/env python3
"""Retrain NAS-DARTS run6 architecture for multi-distance robustness fix.

This script reuses the existing genotype and run6-like retrain configuration,
but uses the multi-distance dataset with augmentation v2 (no horizontal flip)
and optional hand-pair margin loss for cross-hand discrimination.

Usage:
    python retrain_run7_robust.py
    python retrain_run7_robust.py --augmentation-policy v2_multi_distance
    python retrain_run7_robust.py --hand-pair-margin-loss --epochs 100
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "dataset_multi_distance" / "split_info.json"
RUN6_CONFIG_PATH = PROJECT_ROOT / "nas_results" / "retrain_run6" / "config.json"
DEFAULT_GENOTYPE_PATH = PROJECT_ROOT / "nas_results" / "search" / "genotype_final.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "nas_results" / "retrain_run7_robust"


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_split_file(split_path: Path) -> dict[str, Any]:
    """
    Validate and load multi-distance split file.
    
    Expected structure:
    {
        "dataset_root": "dataset_multi_distance",
        "source_folder": "final",
        "subjects": ["835", "836"],
        "label_map": {"835": 0, "836": 1},
        "splits": {
            "train": ["835/final/22cm/image.bmp", ...],
            "val": [...],
            "test": [...]
        },
        "metadata": {
            "train": [{"path": "...", "subject_id": "835", "distance_cm": "22cm"}, ...],
            ...
        }
    }
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    
    split_info = load_json(split_path)
    
    # Validate required keys
    required_keys = ["dataset_root", "source_folder", "subjects", "label_map", "splits"]
    for key in required_keys:
        if key not in split_info:
            raise ValueError(f"Split file missing required key: {key}")
    
    # Validate splits
    for split_name in ["train", "val", "test"]:
        if split_name not in split_info["splits"]:
            raise ValueError(f"Split file missing '{split_name}' split")
        if len(split_info["splits"][split_name]) == 0:
            raise ValueError(f"Split '{split_name}' is empty")
    
    return split_info


def convert_split_to_retrain_format(split_info: dict[str, Any], output_path: Path) -> None:
    """
    Convert multi-distance split format to retrain.py expected format.
    
    retrain.py expects:
    {
        "train": [["subject_id", "filename.bmp"], ...],
        "val": [...],
        "test": [...],
        "subjects": ["835", "836"]
    }
    
    Multi-distance split has:
    {
        "splits": {
            "train": ["835/final/22cm/filename.bmp", ...]
        },
        "metadata": {
            "train": [{"path": "...", "subject_id": "835", ...}, ...]
        }
    }
    """
    retrain_split = {
        "subjects": split_info["subjects"],
        "train": [],
        "val": [],
        "test": []
    }
    
    # Convert each split
    for split_name in ["train", "val", "test"]:
        if "metadata" in split_info and split_name in split_info["metadata"]:
            # Use metadata if available (has subject_id)
            for item in split_info["metadata"][split_name]:
                subject_id = item["subject_id"]
                # Extract filename from path (e.g., "835/final/22cm/image.bmp" -> "image.bmp")
                filename = Path(item["path"]).name
                retrain_split[split_name].append([subject_id, filename])
        else:
            # Fallback: parse from path
            for path_str in split_info["splits"][split_name]:
                # Path format: "835/final/22cm/image.bmp"
                parts = Path(path_str).parts
                subject_id = parts[0]  # "835" or "836"
                filename = parts[-1]   # "image.bmp"
                retrain_split[split_name].append([subject_id, filename])
    
    # Write converted split
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(retrain_split, indent=2), encoding="utf-8")
    
    print(f"Converted split file written to: {output_path}")
    print(f"  Train: {len(retrain_split['train'])} images")
    print(f"  Val:   {len(retrain_split['val'])} images")
    print(f"  Test:  {len(retrain_split['test'])} images")


def build_data_dir_structure(split_info: dict[str, Any], output_dir: Path) -> Path:
    """
    Create symlinks to organize multi-distance images into retrain.py expected structure.
    
    retrain.py expects:
        data_dir/
            835/
                image1.bmp
                image2.bmp
            836/
                image1.bmp
                image2.bmp
    
    Multi-distance has:
        dataset_multi_distance/
            835/final/22cm/image1.bmp
            835/final/25cm/image2.bmp
            ...
    
    We create symlinks to flatten the structure.
    """
    dataset_root = PROJECT_ROOT / split_info["dataset_root"]
    data_dir = output_dir / "data_symlinks"
    
    # Create subject folders
    for subject_id in split_info["subjects"]:
        subject_dir = data_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlinks for all images
    all_paths = []
    for split_name in ["train", "val", "test"]:
        all_paths.extend(split_info["splits"][split_name])
    
    for path_str in all_paths:
        # Path format: "835/final/22cm/image.bmp"
        source_path = dataset_root / path_str
        
        parts = Path(path_str).parts
        subject_id = parts[0]
        filename = parts[-1]
        
        target_path = data_dir / subject_id / filename
        
        # Create symlink if not exists (also check is_symlink to handle broken symlinks)
        if not target_path.exists() and not target_path.is_symlink():
            target_path.symlink_to(source_path.resolve())
    
    print(f"Data directory with symlinks created at: {data_dir}")
    return data_dir


def build_retrain_command(
    args: argparse.Namespace,
    run6_cfg: dict[str, Any],
    split_path: Path,
    data_dir: Path
) -> list[str]:
    """Build command to launch retrain.py with appropriate arguments."""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "retrain.py"),
        "--genotype",
        str(args.genotype),
        "--data_dir",
        str(data_dir),
        "--split_path",
        str(split_path),
        "--output_dir",
        str(args.output_dir),
        "--C_init",
        str(args.C_init if args.C_init is not None else run6_cfg["C_init"]),
        "--num_cells",
        str(args.num_cells if args.num_cells is not None else run6_cfg["num_cells"]),
        "--epochs",
        str(args.epochs if args.epochs is not None else run6_cfg["epochs"]),
        "--batch_size",
        str(args.batch_size if args.batch_size is not None else run6_cfg["batch_size"]),
        "--lr",
        str(args.lr if args.lr is not None else run6_cfg["lr"]),
        "--weight_decay",
        str(args.weight_decay if args.weight_decay is not None else run6_cfg["weight_decay"]),
        "--drop_path_prob",
        str(args.drop_path_prob if args.drop_path_prob is not None else run6_cfg["drop_path_prob"]),
        "--cutout_length",
        str(args.cutout_length if args.cutout_length is not None else run6_cfg["cutout_length"]),
        "--seed",
        str(args.seed if args.seed is not None else run6_cfg["seed"]),
        "--num_workers",
        str(args.num_workers if args.num_workers is not None else run6_cfg["num_workers"]),
    ]
    
    # Auxiliary head
    use_auxiliary = run6_cfg.get("auxiliary", True)
    if args.auxiliary:
        use_auxiliary = True
    if args.no_auxiliary:
        use_auxiliary = False
    command.append("--auxiliary" if use_auxiliary else "--no_auxiliary")
    
    # Augmentation policy (NEW for run7)
    if args.augmentation_policy:
        command.extend(["--augmentation_policy", args.augmentation_policy])
    
    # Hand-pair margin loss (NEW for run7)
    if args.hand_pair_margin_loss:
        command.append("--hand_pair_margin_loss")
        if args.hand_pair_margin is not None:
            command.extend(["--hand_pair_margin", str(args.hand_pair_margin)])
        if args.hand_pair_weight is not None:
            command.extend(["--hand_pair_weight", str(args.hand_pair_weight)])
    
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain NAS-DARTS run6 architecture on multi-distance dataset for robustness fix"
    )
    
    # Dataset and split
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="Path to multi-distance split file (default: dataset_multi_distance/split_info.json)"
    )
    parser.add_argument(
        "--genotype",
        type=Path,
        default=DEFAULT_GENOTYPE_PATH,
        help="Path to genotype JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for training results"
    )
    parser.add_argument(
        "--run6-config",
        type=Path,
        default=RUN6_CONFIG_PATH,
        help="Path to run6 config for baseline hyperparameters"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare split and data directory, don't launch training"
    )
    
    # Augmentation policy (NEW for run7)
    parser.add_argument(
        "--augmentation-policy",
        type=str,
        default="v2_multi_distance",
        choices=["v1_legacy", "v2_multi_distance"],
        help="Augmentation policy: v1_legacy (with horizontal flip) or v2_multi_distance (no flip, more aggressive)"
    )
    
    # Hand-pair margin loss (NEW for run7)
    parser.add_argument(
        "--hand-pair-margin-loss",
        action="store_true",
        help="Enable hand-pair margin loss for cross-hand discrimination"
    )
    parser.add_argument(
        "--hand-pair-margin",
        type=float,
        default=1.0,
        help="Margin for hand-pair loss (default: 1.0)"
    )
    parser.add_argument(
        "--hand-pair-weight",
        type=float,
        default=0.3,
        help="Weight for hand-pair loss (default: 0.3)"
    )
    
    # Optional overrides for retrain.py arguments
    parser.add_argument("--C_init", type=int, default=None)
    parser.add_argument("--num_cells", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--drop_path_prob", type=float, default=None)
    parser.add_argument("--cutout_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--auxiliary", action="store_true")
    parser.add_argument("--no_auxiliary", action="store_true")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load run6 config for baseline hyperparameters
    if not args.run6_config.exists():
        print(f"Warning: run6 config not found at {args.run6_config}")
        print("Using default hyperparameters")
        run6_cfg = {
            "C_init": 16,
            "num_cells": 8,
            "epochs": 100,
            "batch_size": 4,
            "lr": 0.0001,
            "weight_decay": 0.0005,
            "drop_path_prob": 0.2,
            "cutout_length": 16,
            "seed": 42,
            "num_workers": 4,
            "auxiliary": True
        }
    else:
        run6_cfg = load_json(args.run6_config)
    
    # Validate and load split file
    print(f"Loading split file: {args.split_file}")
    split_info = validate_split_file(args.split_file)
    print(f"  Dataset root: {split_info['dataset_root']}")
    print(f"  Source folder: {split_info['source_folder']}")
    print(f"  Subjects: {split_info['subjects']}")
    print(f"  Train: {len(split_info['splits']['train'])} images")
    print(f"  Val:   {len(split_info['splits']['val'])} images")
    print(f"  Test:  {len(split_info['splits']['test'])} images")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert split to retrain.py format
    converted_split_path = args.output_dir / "split_info_converted.json"
    convert_split_to_retrain_format(split_info, converted_split_path)
    
    # Build data directory with symlinks
    data_dir = build_data_dir_structure(split_info, args.output_dir)
    
    # Save run7 config
    run7_config = {
        "base_config": "run6",
        "dataset": "multi_distance",
        "augmentation_policy": args.augmentation_policy,
        "hand_pair_margin_loss": args.hand_pair_margin_loss,
        "hand_pair_margin": args.hand_pair_margin,
        "hand_pair_weight": args.hand_pair_weight,
        "split_file": str(args.split_file),
        "data_dir": str(data_dir),
        "genotype": str(args.genotype),
        **run6_cfg
    }
    
    config_path = args.output_dir / "run7_config.json"
    config_path.write_text(json.dumps(run7_config, indent=2), encoding="utf-8")
    print(f"\nRun7 config saved to: {config_path}")
    
    if args.prepare_only:
        print("\nPrepare-only mode enabled; retrain.py was not launched.")
        print(f"\nTo launch training manually:")
        print(f"  python retrain.py \\")
        print(f"    --genotype {args.genotype} \\")
        print(f"    --data_dir {data_dir} \\")
        print(f"    --split_path {converted_split_path} \\")
        print(f"    --output_dir {args.output_dir} \\")
        print(f"    --augmentation_policy {args.augmentation_policy}")
        if args.hand_pair_margin_loss:
            print(f"    --hand_pair_margin_loss")
        return
    
    # Build and launch retrain command
    command = build_retrain_command(args, run6_cfg, converted_split_path, data_dir)
    
    print("\n" + "=" * 80)
    print("Launching retrain command:")
    print("=" * 80)
    print(" ".join(str(part) for part in command))
    print("=" * 80 + "\n")
    
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

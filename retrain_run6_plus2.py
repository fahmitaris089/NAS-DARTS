"""Retrain NAS-DARTS run6 architecture for the new 2-class final dataset.

This wrapper avoids rerunning architecture search. It reuses the existing
genotype and run6-like retrain configuration, builds a fresh split for the
new dataset under ``captures/final_dataset/preprocessed`` by default, and
then launches ``retrain.py`` with the correct dataset and split paths.

It can also extend an existing split file when ``--base_split_path`` is
explicitly provided.

Usage:
    python retrain_run6_plus2.py
    python retrain_run6_plus2.py --prepare_only
    python retrain_run6_plus2.py --output_dir nas_results/retrain_run6_plus2_e600
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "preprocessed_results"
BASE_SPLIT_PATH: Path | None = None
RUN6_CONFIG_PATH = PROJECT_ROOT / "nas_results" / "retrain_run6" / "config.json"
DEFAULT_GENOTYPE_PATH = PROJECT_ROOT / "nas_results" / "search" / "genotype_final.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "nas_results" / "retrain_run6_plus2"
DEFAULT_SUBJECTS = ("835", "836")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_split_counts(num_images: int) -> tuple[int, int, int]:
    """Choose train/val/test counts while keeping val/test present when possible."""
    if num_images < 3:
        raise ValueError(
            f"Need at least 3 images per new subject, got {num_images}."
        )

    val_count = 1
    test_count = 1
    train_count = num_images - val_count - test_count

    if train_count <= 0:
        raise ValueError(
            f"Split would leave no training images for a class with {num_images} images."
        )
    return train_count, val_count, test_count


def build_split_from_subjects(
    data_dir: Path,
    subjects: list[str],
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build a fresh train/val/test split from the provided subject folders."""
    split = {
        "train": [],
        "val": [],
        "test": [],
        "subjects": [],
    }
    summary: dict[str, dict[str, Any]] = {}

    for offset, subject_id in enumerate(subjects):
        subject_dir = data_dir / subject_id
        if not subject_dir.exists():
            raise FileNotFoundError(f"Missing subject directory: {subject_dir}")

        image_names = sorted(
            path.name for path in subject_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".bmp"
        )
        if not image_names:
            raise ValueError(f"No BMP images found for subject {subject_id} in {subject_dir}")

        train_count, val_count, test_count = infer_split_counts(len(image_names))
        rng = random.Random(seed + offset)
        rng.shuffle(image_names)

        train_names = image_names[:train_count]
        val_names = image_names[train_count:train_count + val_count]
        test_names = image_names[train_count + val_count:train_count + val_count + test_count]

        split["subjects"].append(subject_id)
        split["train"].extend([[subject_id, name] for name in train_names])
        split["val"].extend([[subject_id, name] for name in val_names])
        split["test"].extend([[subject_id, name] for name in test_names])

        summary[subject_id] = {
            "total": len(image_names),
            "train": train_names,
            "val": val_names,
            "test": test_names,
        }

    split["subjects"] = sorted(split["subjects"], key=int)
    return split, summary


def build_extended_split(
    base_split: dict[str, Any],
    data_dir: Path,
    new_subjects: list[str],
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Append new subjects to the original split using a deterministic shuffle."""
    split = {
        "train": [list(item) for item in base_split["train"]],
        "val": [list(item) for item in base_split["val"]],
        "test": [list(item) for item in base_split["test"]],
        "subjects": list(base_split["subjects"]),
    }
    summary: dict[str, dict[str, Any]] = {}

    for offset, subject_id in enumerate(new_subjects):
        if subject_id in split["subjects"]:
            raise ValueError(f"Subject {subject_id} already exists in split file.")

        subject_dir = data_dir / subject_id
        if not subject_dir.exists():
            raise FileNotFoundError(f"Missing subject directory: {subject_dir}")

        image_names = sorted(
            path.name for path in subject_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".bmp"
        )
        if not image_names:
            raise ValueError(f"No BMP images found for subject {subject_id} in {subject_dir}")

        train_count, val_count, test_count = infer_split_counts(len(image_names))
        rng = random.Random(seed + offset)
        rng.shuffle(image_names)

        train_names = image_names[:train_count]
        val_names = image_names[train_count:train_count + val_count]
        test_names = image_names[train_count + val_count:train_count + val_count + test_count]

        split["subjects"].append(subject_id)
        split["train"].extend([[subject_id, name] for name in train_names])
        split["val"].extend([[subject_id, name] for name in val_names])
        split["test"].extend([[subject_id, name] for name in test_names])

        summary[subject_id] = {
            "total": len(image_names),
            "train": train_names,
            "val": val_names,
            "test": test_names,
        }

    split["subjects"] = sorted(split["subjects"], key=int)
    return split, summary


def write_extended_split(
    split_path: Path,
    data_dir: Path,
    subjects: list[str],
    seed: int,
    base_split_path: Path | None,
) -> dict[str, dict[str, Any]]:
    if base_split_path is None:
        final_split, summary = build_split_from_subjects(data_dir, subjects, seed)
    else:
        base_split = load_json(base_split_path)
        final_split, summary = build_extended_split(base_split, data_dir, subjects, seed)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(final_split, indent=2), encoding="utf-8")
    return summary


def build_retrain_command(args: argparse.Namespace, run6_cfg: dict[str, Any], split_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "retrain.py"),
        "--genotype",
        str(args.genotype),
        "--data_dir",
        str(args.data_dir),
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

    use_auxiliary = run6_cfg.get("auxiliary", True)
    if args.auxiliary:
        use_auxiliary = True
    if args.no_auxiliary:
        use_auxiliary = False
    command.append("--auxiliary" if use_auxiliary else "--no_auxiliary")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain NAS-DARTS run6 architecture on the new 2-class final dataset"
    )
    parser.add_argument("--genotype", type=Path, default=DEFAULT_GENOTYPE_PATH)
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--base_split_path", type=Path, default=BASE_SPLIT_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extended_split_path", type=Path, default=None)
    parser.add_argument("--run6_config", type=Path, default=RUN6_CONFIG_PATH)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prepare_only", action="store_true")

    # Optional overrides for retrain.py arguments.
    parser.add_argument("--C_init", type=int, default=None)
    parser.add_argument("--num_cells", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--drop_path_prob", type=float, default=None)
    parser.add_argument("--cutout_length", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--auxiliary", action="store_true")
    parser.add_argument("--no_auxiliary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run6_cfg = load_json(args.run6_config)
    seed = args.seed if args.seed is not None else int(run6_cfg["seed"])

    output_dir = args.output_dir
    split_path = args.extended_split_path or (output_dir / "split_info_836.json")

    summary = write_extended_split(
        split_path=split_path,
        data_dir=args.data_dir,
        subjects=args.subjects,
        seed=seed,
        base_split_path=args.base_split_path,
    )

    print(f"Extended split written to: {split_path}")
    for subject_id, details in summary.items():
        print(
            f"  Subject {subject_id}: total={details['total']} "
            f"train={len(details['train'])} val={len(details['val'])} test={len(details['test'])}"
        )

    if args.prepare_only:
        print("Prepare-only mode enabled; retrain.py was not launched.")
        return

    command = build_retrain_command(args, run6_cfg, split_path)
    print("Launching retrain command:")
    print(" ".join(str(part) for part in command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
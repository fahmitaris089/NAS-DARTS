#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic stratified image-level split")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _allocate_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    raw = [total * train_ratio, total * val_ratio, total * test_ratio]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    if total >= 3 and any(count == 0 for count in counts):
        raise ValueError(f"Ratios produce an empty partition for a class with {total} images: {counts}")
    return counts[0], counts[1], counts[2]


def create_split(
    data_dir: Path,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError(f"Ratios must be positive and sum to 1.0; got {ratios}")

    subjects = sorted(
        (path.name for path in data_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=int,
    )
    if not subjects:
        raise ValueError(f"No numeric subject directories found in {data_dir}")

    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": [], "subjects": subjects}
    expected_per_subject: dict[str, tuple[int, int, int]] = {}
    for subject in subjects:
        subject_dir = data_dir / subject
        images = sorted(path.name for path in subject_dir.glob("*.bmp") if path.is_file())
        if not images:
            raise ValueError(f"Subject {subject} has no BMP images")
        if len(images) != len(set(images)):
            raise ValueError(f"Duplicate image names detected for subject {subject}")
        rng.shuffle(images)
        n_train, n_val, n_test = _allocate_counts(len(images), *ratios)
        expected_per_subject[subject] = (n_train, n_val, n_test)
        split["train"].extend([[subject, name] for name in images[:n_train]])
        split["val"].extend([[subject, name] for name in images[n_train:n_train + n_val]])
        split["test"].extend([[subject, name] for name in images[n_train + n_val:]])

    sets = {name: {(subject, filename) for subject, filename in split[name]} for name in ("train", "val", "test")}
    overlap = {
        "train_val": sets["train"] & sets["val"],
        "train_test": sets["train"] & sets["test"],
        "val_test": sets["val"] & sets["test"],
    }
    if any(overlap.values()):
        raise ValueError(f"Split overlap detected: { {name: len(items) for name, items in overlap.items()} }")

    for index, name in enumerate(("train", "val", "test")):
        counts = Counter(subject for subject, _ in split[name])
        for subject, expected in expected_per_subject.items():
            if counts[subject] != expected[index]:
                raise ValueError(f"Subject {subject} has {counts[subject]} {name} images; expected {expected[index]}")
    return split


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
    split = create_split(
        args.data_dir.resolve(),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "subjects": len(split["subjects"]),
        "counts": {name: len(split[name]) for name in ("train", "val", "test")},
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build split file untuk dataset multi-distance dengan strategi:
1. Campur semua images dari berbagai jarak per subject
2. Split random 60% train / 20% val / 20% test
3. Maintain balance antar subjects (835 dan 836)

Usage:
    python3 build_multi_distance_split.py \
        --dataset-root dataset_multi_distance \
        --output-file dataset_multi_distance/split_info.json \
        --train-ratio 0.6 \
        --val-ratio 0.2 \
        --test-ratio 0.2 \
        --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def collect_images_per_subject(
    dataset_root: Path,
    subject_id: str,
    source_folder: str = "final"
) -> List[Dict]:
    """
    Collect all images dari semua jarak untuk satu subject.
    
    Returns:
        List of dicts with keys: path, subject_id, distance_cm
    """
    subject_folder = dataset_root / subject_id / source_folder
    
    if not subject_folder.exists():
        print(f"Warning: {subject_folder} tidak ditemukan")
        return []
    
    images = []
    
    # Iterate over distance folders
    for distance_folder in sorted(subject_folder.iterdir()):
        if not distance_folder.is_dir():
            continue
        
        distance_cm = distance_folder.name  # e.g., "22cm"
        
        # Collect all BMP images in this distance folder (preprocessed images)
        for img_path in sorted(distance_folder.glob("*.bmp")):
            images.append({
                "path": str(img_path.relative_to(dataset_root)),
                "subject_id": subject_id,
                "distance_cm": distance_cm
            })
    
    return images


def split_images_random(
    images: List[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split images into train/val/test dengan random shuffle.
    
    Returns:
        (train_images, val_images, test_images)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)
    
    # Compute split indices
    n_total = len(shuffled)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val  # Remainder goes to test
    
    # Split
    train_images = shuffled[:n_train]
    val_images = shuffled[n_train:n_train+n_val]
    test_images = shuffled[n_train+n_val:]
    
    return train_images, val_images, test_images


def analyze_split_distribution(
    split_name: str,
    images: List[Dict]
) -> Dict:
    """
    Analyze distribution of images per subject and distance in a split.
    """
    dist_by_subject = defaultdict(lambda: defaultdict(int))
    
    for img in images:
        subject_id = img["subject_id"]
        distance_cm = img["distance_cm"]
        dist_by_subject[subject_id][distance_cm] += 1
    
    return dict(dist_by_subject)


def main():
    parser = argparse.ArgumentParser(
        description="Build split file untuk dataset multi-distance"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root folder dataset (e.g., dataset_multi_distance)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output JSON file untuk split info"
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=["835", "836"],
        help="List of subject IDs (default: 835 836)"
    )
    parser.add_argument(
        "--source-folder",
        type=str,
        default="final",
        help="Source folder name (default: final, for preprocessed BMP images)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Train split ratio (default: 0.6)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    if not args.dataset_root.exists():
        print(f"Error: Dataset root {args.dataset_root} tidak ditemukan")
        sys.exit(1)
    
    print(f"Building split file from {args.dataset_root}")
    print(f"Subjects: {args.subjects}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)
    
    # Collect images per subject
    all_images_by_subject = {}
    
    for subject_id in args.subjects:
        print(f"\nCollecting images for subject {subject_id}...")
        images = collect_images_per_subject(
            args.dataset_root,
            subject_id,
            args.source_folder
        )
        
        if len(images) == 0:
            print(f"  ⚠️ No images found for subject {subject_id}")
            continue
        
        print(f"  ✓ Found {len(images)} images")
        
        # Count per distance
        dist_counts = defaultdict(int)
        for img in images:
            dist_counts[img["distance_cm"]] += 1
        
        for distance_cm, count in sorted(dist_counts.items()):
            print(f"    - {distance_cm}: {count} images")
        
        all_images_by_subject[subject_id] = images
    
    # Combine all images
    all_images = []
    for subject_id, images in all_images_by_subject.items():
        all_images.extend(images)
    
    print(f"\n{'=' * 80}")
    print(f"Total images: {len(all_images)}")
    
    # Split images
    print(f"\nSplitting images...")
    train_images, val_images, test_images = split_images_random(
        all_images,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    print(f"  Train: {len(train_images)} images ({len(train_images)/len(all_images)*100:.1f}%)")
    print(f"  Val:   {len(val_images)} images ({len(val_images)/len(all_images)*100:.1f}%)")
    print(f"  Test:  {len(test_images)} images ({len(test_images)/len(all_images)*100:.1f}%)")
    
    # Analyze distribution per split
    print(f"\n{'=' * 80}")
    print("DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    for split_name, split_images in [
        ("Train", train_images),
        ("Val", val_images),
        ("Test", test_images)
    ]:
        print(f"\n{split_name} ({len(split_images)} images):")
        dist = analyze_split_distribution(split_name, split_images)
        
        for subject_id in sorted(dist.keys()):
            subject_dist = dist[subject_id]
            total = sum(subject_dist.values())
            print(f"  Subject {subject_id} ({total} images):")
            for distance_cm in sorted(subject_dist.keys()):
                count = subject_dist[distance_cm]
                print(f"    - {distance_cm}: {count} images")
    
    # Build split info structure
    split_info = {
        "dataset_root": str(args.dataset_root),
        "source_folder": args.source_folder,
        "subjects": args.subjects,
        "label_map": {subject_id: i for i, subject_id in enumerate(args.subjects)},
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio
        },
        "random_seed": args.seed,
        "splits": {
            "train": [img["path"] for img in train_images],
            "val": [img["path"] for img in val_images],
            "test": [img["path"] for img in test_images]
        },
        "metadata": {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }
    }
    
    # Save to JSON
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"✓ Split file saved to {args.output_file}")
    print(f"✓ Total: {len(all_images)} images")
    print(f"  - Train: {len(train_images)} ({len(train_images)/len(all_images)*100:.1f}%)")
    print(f"  - Val:   {len(val_images)} ({len(val_images)/len(all_images)*100:.1f}%)")
    print(f"  - Test:  {len(test_images)} ({len(test_images)/len(all_images)*100:.1f}%)")
    
    # Summary recommendations
    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Check balance
    train_dist = analyze_split_distribution("Train", train_images)
    imbalance = []
    
    for subject_id in args.subjects:
        if subject_id in train_dist:
            subject_total = sum(train_dist[subject_id].values())
            expected = len(train_images) / len(args.subjects)
            if abs(subject_total - expected) > 2:
                imbalance.append(f"Subject {subject_id}: {subject_total} (expected ~{expected:.0f})")
    
    if imbalance:
        print("\n⚠️ Class imbalance detected in training set:")
        for msg in imbalance:
            print(f"  - {msg}")
        print("  Consider re-running with different seed or adjusting dataset")
    else:
        print("\n✅ Training set is well-balanced across subjects")
    
    # Check distance coverage
    min_distance_coverage = float('inf')
    for split_name, split_images in [("Train", train_images), ("Val", val_images), ("Test", test_images)]:
        dist = analyze_split_distribution(split_name, split_images)
        for subject_id in dist:
            for distance_cm in dist[subject_id]:
                count = dist[subject_id][distance_cm]
                min_distance_coverage = min(min_distance_coverage, count)
    
    if min_distance_coverage == 0:
        print("\n⚠️ Some (subject, distance) combinations have 0 samples in a split")
        print("  This may affect robustness evaluation")
    elif min_distance_coverage < 2:
        print(f"\n⚠️ Minimum coverage per (subject, distance): {min_distance_coverage} samples")
        print("  Consider acquiring more data for better coverage")
    else:
        print(f"\n✅ Good distance coverage: minimum {min_distance_coverage} samples per (subject, distance)")


if __name__ == "__main__":
    main()

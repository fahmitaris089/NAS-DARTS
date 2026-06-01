#!/usr/bin/env python3
"""
Script untuk memilih 10 raw images terbaik dari tiap jarak berdasarkan laplacian variance.

Usage:
    python3 select_best_raw_images.py \
        --dataset-root dataset_multi_distance/835 \
        --output-dir dataset_multi_distance/835/final_raw \
        --samples-per-distance 10
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import json


def compute_laplacian_variance(image_path: Path) -> float:
    """Compute Laplacian variance sebagai proxy sharpness/quality."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return float(laplacian.var())


def select_best_images(
    raw_folder: Path,
    n_samples: int
) -> List[Tuple[Path, float]]:
    """
    Pilih n_samples images terbaik dari raw folder berdasarkan laplacian variance.
    
    Returns:
        List of (image_path, laplacian_var) tuples, sorted by quality (best first)
    """
    if not raw_folder.exists():
        print(f"Warning: {raw_folder} tidak ditemukan")
        return []
    
    # Collect all raw images with their quality scores
    image_scores = []
    
    for img_path in raw_folder.glob("*.png"):
        lap_var = compute_laplacian_variance(img_path)
        image_scores.append((img_path, lap_var))
    
    if len(image_scores) == 0:
        print(f"Warning: Tidak ada images di {raw_folder}")
        return []
    
    # Sort by laplacian variance (descending = best first)
    image_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top n_samples
    selected = image_scores[:n_samples]
    
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Pilih raw images terbaik dari tiap jarak berdasarkan quality metrics"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root folder dataset (e.g., dataset_multi_distance/835)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory untuk final raw images"
    )
    parser.add_argument(
        "--samples-per-distance",
        type=int,
        default=10,
        help="Jumlah samples yang dipilih per jarak (default: 10)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hanya print selection tanpa copy files"
    )
    
    args = parser.parse_args()
    
    if not args.dataset_root.exists():
        print(f"Error: Dataset root {args.dataset_root} tidak ditemukan")
        sys.exit(1)
    
    # Create output directory structure
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Selecting {args.samples_per_distance} best raw images per distance...")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)
    
    # Track selection results
    selection_summary = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "samples_per_distance": args.samples_per_distance,
        "distances": {}
    }
    
    total_selected = 0
    
    # Process each distance folder
    for distance_folder in sorted(args.dataset_root.iterdir()):
        if not distance_folder.is_dir():
            continue
        
        distance_cm = distance_folder.name
        raw_folder = distance_folder / "raw"
        
        print(f"\n{distance_cm}:")
        print("-" * 40)
        
        if not raw_folder.exists():
            print(f"  ⚠️ Raw folder tidak ditemukan: {raw_folder}")
            continue
        
        # Count available images
        available_images = list(raw_folder.glob("*.png"))
        print(f"  Available: {len(available_images)} images")
        
        # Select best images
        selected = select_best_images(raw_folder, args.samples_per_distance)
        
        if len(selected) == 0:
            print(f"  ⚠️ Tidak ada images yang bisa dipilih")
            continue
        
        print(f"  Selected: {len(selected)} images (target: {args.samples_per_distance})")
        
        # Create distance subfolder in output
        output_distance_dir = args.output_dir / distance_cm
        if not args.dry_run:
            output_distance_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy selected images
        selected_files = []
        for i, (img_path, lap_var) in enumerate(selected, 1):
            dest_path = output_distance_dir / img_path.name
            
            if not args.dry_run:
                shutil.copy2(img_path, dest_path)
            
            selected_files.append({
                "filename": img_path.name,
                "laplacian_var": lap_var,
                "rank": i
            })
            
            print(f"    {i:2d}. {img_path.name} (lap_var: {lap_var:.1f})")
        
        # Store summary
        selection_summary["distances"][distance_cm] = {
            "available": len(available_images),
            "selected": len(selected),
            "target": args.samples_per_distance,
            "selected_files": selected_files,
            "quality_stats": {
                "mean": float(np.mean([s[1] for s in selected])),
                "std": float(np.std([s[1] for s in selected])),
                "min": float(np.min([s[1] for s in selected])),
                "max": float(np.max([s[1] for s in selected]))
            }
        }
        
        total_selected += len(selected)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for distance_cm, info in sorted(selection_summary["distances"].items()):
        status = "✅" if info["selected"] == info["target"] else "⚠️"
        print(f"{status} {distance_cm}: {info['selected']}/{info['target']} samples")
        print(f"   Quality: {info['quality_stats']['mean']:.1f} ± {info['quality_stats']['std']:.1f}")
    
    print(f"\nTotal selected: {total_selected} images")
    
    # Save selection summary to JSON
    if not args.dry_run:
        summary_path = args.output_dir / "selection_summary.json"
        with open(summary_path, "w") as f:
            json.dump(selection_summary, f, indent=2)
        print(f"\n✓ Selection summary saved to {summary_path}")
        print(f"✓ Raw images copied to {args.output_dir}/")
    else:
        print("\n[DRY RUN] No files were copied")
    
    # Check if we need to acquire more samples
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    shortfall_distances = []
    for distance_cm, info in sorted(selection_summary["distances"].items()):
        if info["selected"] < info["target"]:
            shortfall = info["target"] - info["selected"]
            shortfall_distances.append((distance_cm, shortfall))
    
    if shortfall_distances:
        print("\n⚠️ Beberapa jarak memerlukan tambahan samples:")
        for distance_cm, shortfall in shortfall_distances:
            print(f"   - {distance_cm}: butuh +{shortfall} samples")
        print("\nAction: Akuisisi tambahan samples di jarak tersebut")
    else:
        print("\n✅ Semua jarak sudah memenuhi target!")
        print(f"   Total: {total_selected} samples")
        print(f"   Ready untuk preprocessing dan training")


if __name__ == "__main__":
    main()

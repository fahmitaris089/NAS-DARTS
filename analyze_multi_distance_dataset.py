#!/usr/bin/env python3
"""
Analisis kualitas dataset multi-distance untuk identifikasi bottleneck robustness.

Script ini menganalisis:
1. Volume data per jarak (apakah 10 sample cukup?)
2. Kualitas image (laplacian variance, contrast, brightness)
3. Konsistensi ROI size (palm_core_side_px) sebagai proxy jarak
4. Distribusi embedding space (jika model tersedia)
5. Rekomendasi: apakah perlu tambahan sample di jarak kritis

Usage:
    python3 analyze_multi_distance_dataset.py \
        --dataset-root dataset_multi_distance/835 \
        --output-dir analysis_results \
        --model-path nas_results/retrain_run6_plus2_e100/best_model.pth \
        --onnx-path nas_results/retrain_run6_plus2_e100/retrain_run6_plus2.onnx
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Import preprocessing untuk re-compute quality metrics
try:
    from palm_preprocessing import preprocess_palm_image
except ImportError:
    print("Warning: palm_preprocessing not found, quality metrics will be limited")
    preprocess_palm_image = None


def compute_laplacian_variance(image_path: Path) -> float:
    """Compute Laplacian variance sebagai proxy sharpness/quality."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return float(laplacian.var())


def compute_image_stats(image_path: Path) -> Dict:
    """Compute basic image statistics."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    
    return {
        "mean": float(np.mean(img)),
        "std": float(np.std(img)),
        "min": int(np.min(img)),
        "max": int(np.max(img)),
        "shape": img.shape
    }


def analyze_distance_folder(distance_folder: Path) -> Dict:
    """Analisis satu folder jarak (e.g., 22cm/)."""
    final_folder = distance_folder / "final"
    
    if not final_folder.exists():
        return {
            "distance_cm": distance_folder.name,
            "num_samples": 0,
            "error": "final folder not found"
        }
    
    final_images = sorted(final_folder.glob("*.png"))
    
    if len(final_images) == 0:
        return {
            "distance_cm": distance_folder.name,
            "num_samples": 0,
            "error": "no final images found"
        }
    
    # Compute quality metrics untuk setiap image
    laplacian_vars = []
    image_stats_list = []
    
    for img_path in final_images:
        lap_var = compute_laplacian_variance(img_path)
        laplacian_vars.append(lap_var)
        
        stats = compute_image_stats(img_path)
        image_stats_list.append(stats)
    
    # Aggregate statistics
    result = {
        "distance_cm": distance_folder.name,
        "num_samples": len(final_images),
        "laplacian_variance": {
            "mean": float(np.mean(laplacian_vars)),
            "std": float(np.std(laplacian_vars)),
            "min": float(np.min(laplacian_vars)),
            "max": float(np.max(laplacian_vars)),
            "samples": laplacian_vars
        },
        "brightness": {
            "mean": float(np.mean([s["mean"] for s in image_stats_list])),
            "std": float(np.std([s["mean"] for s in image_stats_list]))
        },
        "contrast": {
            "mean": float(np.mean([s["std"] for s in image_stats_list])),
            "std": float(np.std([s["std"] for s in image_stats_list]))
        },
        "image_paths": [str(p) for p in final_images]
    }
    
    return result


def analyze_roi_consistency(dataset_root: Path) -> Dict:
    """
    Analisis konsistensi ROI size (palm_core_side_px) sebagai proxy jarak.
    Ini penting untuk OOD detector (M-4).
    """
    if preprocess_palm_image is None:
        return {"error": "palm_preprocessing not available"}
    
    roi_sizes_by_distance = defaultdict(list)
    
    for distance_folder in sorted(dataset_root.iterdir()):
        if not distance_folder.is_dir():
            continue
        
        distance_cm = distance_folder.name
        final_folder = distance_folder / "final"
        
        if not final_folder.exists():
            continue
        
        for img_path in final_folder.glob("*.png"):
            try:
                # Re-run preprocessing untuk extract ROI size
                result = preprocess_palm_image(
                    str(img_path),
                    profile="dataset_v3",
                    save_debug=False
                )
                
                if result and "debug" in result and "roi_side" in result["debug"]:
                    roi_side_px = result["debug"]["roi_side"]
                    roi_sizes_by_distance[distance_cm].append(roi_side_px)
            except Exception as e:
                print(f"Warning: failed to preprocess {img_path}: {e}")
                continue
    
    # Compute statistics per distance
    roi_stats = {}
    for distance_cm, sizes in roi_sizes_by_distance.items():
        if len(sizes) > 0:
            roi_stats[distance_cm] = {
                "mean": float(np.mean(sizes)),
                "std": float(np.std(sizes)),
                "min": int(np.min(sizes)),
                "max": int(np.max(sizes)),
                "samples": sizes
            }
    
    return roi_stats


def plot_quality_distribution(analysis_results: List[Dict], output_dir: Path):
    """Plot distribusi quality metrics per jarak."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    distances = []
    lap_vars_mean = []
    lap_vars_std = []
    num_samples = []
    
    for result in analysis_results:
        if "error" in result:
            continue
        distances.append(result["distance_cm"])
        lap_vars_mean.append(result["laplacian_variance"]["mean"])
        lap_vars_std.append(result["laplacian_variance"]["std"])
        num_samples.append(result["num_samples"])
    
    # Plot 1: Laplacian variance per distance
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].bar(distances, lap_vars_mean, yerr=lap_vars_std, capsize=5)
    axes[0].axhline(y=60, color='r', linestyle='--', label='Quality threshold (60)')
    axes[0].set_xlabel('Distance (cm)')
    axes[0].set_ylabel('Laplacian Variance (mean ± std)')
    axes[0].set_title('Image Sharpness per Distance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Number of samples per distance
    axes[1].bar(distances, num_samples)
    axes[1].axhline(y=10, color='g', linestyle='--', label='Current target (10)')
    axes[1].axhline(y=25, color='orange', linestyle='--', label='Original target (25)')
    axes[1].set_xlabel('Distance (cm)')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Dataset Volume per Distance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "quality_distribution.png", dpi=150)
    print(f"Saved quality distribution plot to {output_dir / 'quality_distribution.png'}")
    plt.close()


def plot_roi_size_distribution(roi_stats: Dict, output_dir: Path):
    """Plot distribusi ROI size (palm_core_side_px) per jarak."""
    if not roi_stats or "error" in roi_stats:
        print("Skipping ROI size plot (preprocessing not available)")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    distances = []
    roi_means = []
    roi_stds = []
    
    for distance_cm in sorted(roi_stats.keys()):
        distances.append(distance_cm)
        roi_means.append(roi_stats[distance_cm]["mean"])
        roi_stds.append(roi_stats[distance_cm]["std"])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(distances, roi_means, yerr=roi_stds, marker='o', capsize=5, linewidth=2)
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('ROI Size (palm_core_side_px)')
    ax.set_title('ROI Size Distribution per Distance\n(Proxy for OOD Detection)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "roi_size_distribution.png", dpi=150)
    print(f"Saved ROI size distribution plot to {output_dir / 'roi_size_distribution.png'}")
    plt.close()


def generate_recommendations(analysis_results: List[Dict], roi_stats: Dict) -> Dict:
    """
    Generate rekomendasi berdasarkan analisis:
    1. Apakah 10 sample cukup?
    2. Apakah perlu tambahan sample di jarak kritis?
    3. Apakah quality metrics memenuhi threshold?
    """
    recommendations = {
        "summary": "",
        "volume_assessment": "",
        "quality_assessment": "",
        "critical_distances": [],
        "action_items": []
    }
    
    # Volume assessment
    total_samples = sum(r["num_samples"] for r in analysis_results if "error" not in r)
    num_distances = len([r for r in analysis_results if "error" not in r])
    
    if total_samples < 50:
        recommendations["volume_assessment"] = (
            f"⚠️ INSUFFICIENT: Total {total_samples} samples across {num_distances} distances. "
            f"Target minimum adalah 50 samples (10 per distance × 5 distances). "
            f"Dengan volume ini, model akan struggle untuk generalisasi."
        )
        recommendations["action_items"].append(
            "CRITICAL: Tambahkan minimal 5 sample per jarak untuk mencapai 75 total samples."
        )
    elif total_samples < 75:
        recommendations["volume_assessment"] = (
            f"⚠️ MARGINAL: Total {total_samples} samples. Ini adalah minimum absolut. "
            f"Model akan memiliki robustness terbatas. Target ideal adalah 125 samples (25 per distance)."
        )
        recommendations["action_items"].append(
            "RECOMMENDED: Tambahkan 10-15 sample per jarak untuk mencapai 100-125 total samples."
        )
    else:
        recommendations["volume_assessment"] = (
            f"✅ ACCEPTABLE: Total {total_samples} samples. Volume ini cukup untuk training awal."
        )
    
    # Quality assessment
    low_quality_distances = []
    for result in analysis_results:
        if "error" in result:
            continue
        
        lap_var_mean = result["laplacian_variance"]["mean"]
        if lap_var_mean < 60:
            low_quality_distances.append({
                "distance": result["distance_cm"],
                "lap_var": lap_var_mean
            })
    
    if low_quality_distances:
        recommendations["quality_assessment"] = (
            f"⚠️ LOW QUALITY: {len(low_quality_distances)} distance(s) memiliki laplacian variance < 60. "
            f"Images ini mungkin terlalu blur dan akan di-reject oleh quality filter."
        )
        for item in low_quality_distances:
            recommendations["action_items"].append(
                f"Re-capture images di {item['distance']} dengan focus yang lebih baik (current lap_var: {item['lap_var']:.1f})"
            )
    else:
        recommendations["quality_assessment"] = (
            "✅ QUALITY OK: Semua distances memiliki laplacian variance ≥ 60."
        )
    
    # Critical distances (ekstrem: 22cm dan 32cm)
    critical_distances_data = [r for r in analysis_results if r.get("distance_cm") in ["22cm", "32cm"]]
    
    for result in critical_distances_data:
        if result["num_samples"] < 15:
            recommendations["critical_distances"].append({
                "distance": result["distance_cm"],
                "current_samples": result["num_samples"],
                "recommended_samples": 15,
                "reason": "Jarak ekstrem memerlukan lebih banyak sample untuk robustness"
            })
    
    if recommendations["critical_distances"]:
        recommendations["action_items"].append(
            f"PRIORITY: Tambahkan 5 sample di jarak kritis: {', '.join([d['distance'] for d in recommendations['critical_distances']])}"
        )
    
    # ROI consistency check
    if roi_stats and "error" not in roi_stats:
        roi_cv_by_distance = {}
        for distance_cm, stats in roi_stats.items():
            cv = stats["std"] / stats["mean"] if stats["mean"] > 0 else 0
            roi_cv_by_distance[distance_cm] = cv
        
        high_variance_distances = [d for d, cv in roi_cv_by_distance.items() if cv > 0.15]
        
        if high_variance_distances:
            recommendations["action_items"].append(
                f"⚠️ HIGH ROI VARIANCE: Distances {', '.join(high_variance_distances)} memiliki CV > 15%. "
                f"Ini menunjukkan inconsistent hand positioning. Pastikan jarak tangan ke kamera konsisten saat capture."
            )
    
    # Summary
    if len(recommendations["action_items"]) == 0:
        recommendations["summary"] = (
            "✅ Dataset quality ACCEPTABLE untuk training awal. "
            "Lanjutkan ke Task 7 (retrain dengan augmentation v2)."
        )
    else:
        recommendations["summary"] = (
            f"⚠️ Dataset memerlukan perbaikan sebelum training. "
            f"Total {len(recommendations['action_items'])} action items."
        )
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Analisis dataset multi-distance untuk robustness fix")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Root folder dataset (e.g., dataset_multi_distance/835)")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results"),
                        help="Output directory untuk hasil analisis")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Path ke model .pth untuk embedding analysis (optional)")
    parser.add_argument("--onnx-path", type=Path, default=None,
                        help="Path ke ONNX model untuk embedding analysis (optional)")
    
    args = parser.parse_args()
    
    if not args.dataset_root.exists():
        print(f"Error: Dataset root {args.dataset_root} tidak ditemukan")
        sys.exit(1)
    
    print(f"Analyzing dataset: {args.dataset_root}")
    print("=" * 80)
    
    # Analisis per distance folder
    analysis_results = []
    for distance_folder in sorted(args.dataset_root.iterdir()):
        if not distance_folder.is_dir():
            continue
        
        print(f"\nAnalyzing {distance_folder.name}...")
        result = analyze_distance_folder(distance_folder)
        analysis_results.append(result)
        
        if "error" in result:
            print(f"  ⚠️ Error: {result['error']}")
        else:
            print(f"  ✓ Samples: {result['num_samples']}")
            print(f"  ✓ Laplacian variance: {result['laplacian_variance']['mean']:.1f} ± {result['laplacian_variance']['std']:.1f}")
            print(f"  ✓ Brightness: {result['brightness']['mean']:.1f} ± {result['brightness']['std']:.1f}")
    
    # Analisis ROI consistency
    print("\n" + "=" * 80)
    print("Analyzing ROI size consistency (proxy for distance)...")
    roi_stats = analyze_roi_consistency(args.dataset_root)
    
    if "error" not in roi_stats:
        for distance_cm, stats in sorted(roi_stats.items()):
            print(f"  {distance_cm}: ROI size = {stats['mean']:.1f} ± {stats['std']:.1f} px")
    
    # Generate plots
    print("\n" + "=" * 80)
    print("Generating plots...")
    plot_quality_distribution(analysis_results, args.output_dir)
    plot_roi_size_distribution(roi_stats, args.output_dir)
    
    # Generate recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    recommendations = generate_recommendations(analysis_results, roi_stats)
    
    print(f"\n{recommendations['summary']}\n")
    print(f"Volume: {recommendations['volume_assessment']}\n")
    print(f"Quality: {recommendations['quality_assessment']}\n")
    
    if recommendations["critical_distances"]:
        print("Critical Distances:")
        for item in recommendations["critical_distances"]:
            print(f"  - {item['distance']}: {item['current_samples']} samples (recommended: {item['recommended_samples']})")
            print(f"    Reason: {item['reason']}")
        print()
    
    if recommendations["action_items"]:
        print("Action Items:")
        for i, action in enumerate(recommendations["action_items"], 1):
            print(f"  {i}. {action}")
    
    # Save results to JSON
    output_json = args.output_dir / "analysis_results.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, "w") as f:
        json.dump({
            "dataset_root": str(args.dataset_root),
            "analysis_results": analysis_results,
            "roi_stats": roi_stats,
            "recommendations": recommendations
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_json}")
    print(f"✓ Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

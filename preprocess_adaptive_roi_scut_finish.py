"""Adaptive-ROI + SCUT-style finishing comparison script.

This uses the adaptive ROI logic from palm_preprocessing.py, but applies a
SCUT-like finalization stage:
- no denoise
- CLAHE clip=2.0 tile=8x8
- min-max normalization
- Lanczos resize to 224x224
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from palm_preprocessing import (
    PalmPreprocessingConfig,
    apply_clahe,
    extract_adaptive_roi,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "captures/res_1920x1080_dataset_v3/raw"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "captures/res_1920x1080_dataset_v3/adaptive_roi_scut_finish"
DEFAULT_COMPARE_DIR = ROOT_DIR / "captures/res_1920x1080_dataset_v3/final"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def default_config() -> PalmPreprocessingConfig:
    return PalmPreprocessingConfig(
        roi_size=820,
        final_size=224,
        clahe_clip=2.0,
        clahe_tile=(8, 8),
        centroid_window=0,
        denoise_h=0.0,
        adaptive_roi=True,
        adaptive_roi_scale=0.90,
        palm_core_width_ratio=0.60,
        center_offset_x=0,
        center_offset_y=20,
    )


def image_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalize_and_resize_scut(gray: np.ndarray, final_size: int) -> np.ndarray:
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.resize(
        normalized,
        (final_size, final_size),
        interpolation=cv2.INTER_LANCZOS4,
    )


def preprocess_gray(gray: np.ndarray, config: PalmPreprocessingConfig) -> dict:
    roi, debug = extract_adaptive_roi(
        gray,
        roi_scale=float(config.adaptive_roi_scale),
        width_ratio=float(config.palm_core_width_ratio),
        centroid_window=int(config.centroid_window),
        center_offset=(int(config.center_offset_x), int(config.center_offset_y)),
    )
    clahe = apply_clahe(roi, config.clahe_clip, config.clahe_tile)
    final = normalize_and_resize_scut(clahe, config.final_size)
    return {
        "raw": gray,
        "roi": roi,
        "clahe": clahe,
        "final": final,
        "debug": debug,
    }


def write_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"Failed to write image: {path}")


def visualize_compare(
    path: Path,
    stem: str,
    result: dict,
    baseline_final: np.ndarray | None,
) -> None:
    raw = result["raw"]
    roi = result["roi"]
    clahe = result["clahe"]
    final = result["final"]
    debug = result["debug"]
    mask = debug["palm_mask"]

    if baseline_final is None:
        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    else:
        fig, axes = plt.subplots(1, 6, figsize=(26, 5))
    fig.suptitle(f"Adaptive ROI + SCUT finish: {stem}", fontsize=12)

    ax = axes[0]
    ax.imshow(raw, cmap="gray")
    ref = debug["refined_center"]
    box = debug["roi_box"]
    ax.plot(*ref, "r+", markersize=12, markeredgewidth=2)
    ax.add_patch(
        plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
    )
    ax.set_title("1. Raw")
    ax.axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("2. Palm Mask")
    axes[1].axis("off")

    axes[2].imshow(roi, cmap="gray")
    axes[2].set_title(f"3. Adaptive ROI\n{roi.shape[1]}x{roi.shape[0]}")
    axes[2].axis("off")

    axes[3].imshow(clahe, cmap="gray")
    axes[3].set_title("4. CLAHE")
    axes[3].axis("off")

    axes[4].imshow(final, cmap="gray")
    axes[4].set_title("5. SCUT-like Final")
    axes[4].axis("off")

    if baseline_final is not None:
        axes[5].imshow(baseline_final, cmap="gray")
        axes[5].set_title("6. Current Final")
        axes[5].axis("off")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive ROI with SCUT-style finishing for raw capture data.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--no-skip", action="store_true", help="Reprocess existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = image_files(args.input_dir)
    if not images:
        raise SystemExit(f"No images found in: {args.input_dir}")

    config = default_config()
    output_dir = args.output_dir
    final_dir = output_dir / "final"
    roi_dir = output_dir / "roi"
    clahe_dir = output_dir / "clahe"
    viz_dir = output_dir / "visualizations"

    print(f"Input  : {args.input_dir}")
    print(f"Output : {output_dir}")
    print(f"Images : {len(images)}")
    print("-" * 60)

    processed = 0
    skipped = 0
    for index, img_path in enumerate(images, start=1):
        stem = img_path.stem
        out_final = final_dir / f"{stem}_final.png"
        if out_final.exists() and not args.no_skip:
            skipped += 1
            print(f"[{index:>3}/{len(images)}] {img_path.name} SKIP")
            continue

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[{index:>3}/{len(images)}] {img_path.name} FAIL read")
            continue

        result = preprocess_gray(gray, config)
        write_gray(out_final, result["final"])
        write_gray(roi_dir / f"{stem}_roi.png", result["roi"])
        write_gray(clahe_dir / f"{stem}_clahe.png", result["clahe"])

        baseline_path = args.compare_dir / f"{stem}_final.png"
        baseline_final = None
        if baseline_path.exists():
            baseline_final = cv2.imread(str(baseline_path), cv2.IMREAD_GRAYSCALE)

        visualize_compare(viz_dir / f"{stem}_compare.png", stem, result, baseline_final)

        processed += 1
        print(f"[{index:>3}/{len(images)}] {img_path.name} OK")

    print("=" * 60)
    print(f"Finished. processed={processed}, skipped={skipped}")
    print(f"Final images: {final_dir}")


if __name__ == "__main__":
    main()

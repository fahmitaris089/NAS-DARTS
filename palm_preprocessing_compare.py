"""Batch comparison preprocessing for palm-vein raw captures.

This script is meant for experimentation only. It keeps the current baseline
pipeline and adds an alternative branch that suppresses broad palm lines while
trying to preserve thinner vein structures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from palm_preprocessing import (
    PROFILE_DATASET_V3,
    PalmPreprocessingConfig,
    percentile_stretch,
    preprocess_palm_image,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "captures/res_1920x1080_dataset_v3/raw"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "captures/res_1920x1080_dataset_v3/compare_linesuppressed"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def default_config() -> PalmPreprocessingConfig:
    return PalmPreprocessingConfig(
        roi_size=820,
        final_size=224,
        clahe_clip=2.0,
        clahe_tile=(8, 8),
        centroid_window=0,
        profile=PROFILE_DATASET_V3,
        denoise_h=3.0,
        vessel_preview_kernel=17,
        center_offset_x=0,
        center_offset_y=20,
        stretch_percentiles=(1.0, 99.5),
        adaptive_roi=True,
        adaptive_roi_scale=0.90,
        palm_core_width_ratio=0.60,
    )


def image_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def roi_mask_from_result(result: Dict[str, object]) -> np.ndarray:
    debug = result["debug"]
    x1, y1, x2, y2 = debug["roi_box"]
    full_mask = result["mask"]
    roi_mask = full_mask[y1:y2, x1:x2]
    if roi_mask.shape != result["roi"].shape:
        roi_mask = cv2.resize(
            roi_mask,
            (result["roi"].shape[1], result["roi"].shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return roi_mask


def max_blackhat(gray: np.ndarray, kernels: Tuple[int, ...]) -> np.ndarray:
    responses = []
    for kernel_size in kernels:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        responses.append(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))
    return np.maximum.reduce(responses)


def suppress_palm_lines(
    clahe: np.ndarray,
    roi_mask: np.ndarray,
    final_size: int,
) -> Dict[str, np.ndarray]:
    """Favor thin vein-like structures over broader palm creases."""
    smoothed = cv2.fastNlMeansDenoising(
        clahe,
        None,
        h=4.0,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    small_dark = max_blackhat(smoothed, (9, 13, 17))
    large_dark = max_blackhat(smoothed, (25, 31, 41))

    small_dark = cv2.bitwise_and(small_dark, small_dark, mask=roi_mask)
    large_dark = cv2.bitwise_and(large_dark, large_dark, mask=roi_mask)

    vessel_response = cv2.subtract(
        small_dark,
        cv2.convertScaleAbs(large_dark, alpha=0.70),
    )
    vessel_response = percentile_stretch(vessel_response, 2.0, 99.8)

    smooth_base = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=1.4)
    line_suppressed = smooth_base.astype(np.float32) - (
        1.05 * vessel_response.astype(np.float32)
    )
    line_suppressed = np.clip(line_suppressed, 0, 255).astype(np.uint8)
    line_suppressed = percentile_stretch(line_suppressed, 1.0, 99.5)
    final = cv2.resize(
        line_suppressed,
        (final_size, final_size),
        interpolation=cv2.INTER_AREA,
    )

    return {
        "smoothed": smoothed,
        "small_dark": small_dark,
        "large_dark": large_dark,
        "vessel_response": vessel_response,
        "line_suppressed_source": line_suppressed,
        "final": final,
    }


def write_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"Failed to write image: {path}")


def visualize_comparison(
    path: Path,
    result: Dict[str, object],
    compare: Dict[str, np.ndarray],
) -> None:
    raw = result["raw"]
    roi = result["roi"]
    clahe = result["clahe"]
    baseline = result["final"]
    vessel_response = compare["vessel_response"]
    suppressed = compare["final"]

    fig, axes = plt.subplots(1, 6, figsize=(26, 5))
    fig.suptitle(path.name, fontsize=12)

    axes[0].imshow(raw, cmap="gray")
    axes[0].set_title("1. Raw")
    axes[0].axis("off")

    axes[1].imshow(roi, cmap="gray")
    axes[1].set_title("2. ROI")
    axes[1].axis("off")

    axes[2].imshow(clahe, cmap="gray")
    axes[2].set_title("3. CLAHE")
    axes[2].axis("off")

    axes[3].imshow(vessel_response, cmap="gray")
    axes[3].set_title("4. Thin-Vein Response")
    axes[3].axis("off")

    axes[4].imshow(baseline, cmap="gray")
    axes[4].set_title("5. Baseline Final")
    axes[4].axis("off")

    axes[5].imshow(suppressed, cmap="gray")
    axes[5].set_title("6. Line-Suppressed Final")
    axes[5].axis("off")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline preprocessing with palm-line-suppressed preprocessing.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images.")
    parser.add_argument("--no-skip", action="store_true", help="Reprocess existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_config()
    images = image_files(args.input_dir)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in: {args.input_dir}")

    output_dir = args.output_dir
    baseline_dir = output_dir / "baseline_final"
    compare_dir = output_dir / "line_suppressed_final"
    vessel_dir = output_dir / "thin_vein_response"
    viz_dir = output_dir / "visualizations"

    print(f"Input  : {args.input_dir}")
    print(f"Output : {output_dir}")
    print(f"Images : {len(images)}")
    print("-" * 60)

    processed = 0
    skipped = 0
    for index, img_path in enumerate(images, start=1):
        stem = img_path.stem
        compare_path = compare_dir / f"{stem}_final.png"
        if compare_path.exists() and not args.no_skip:
            skipped += 1
            print(f"[{index:>3}/{len(images)}] {img_path.name} SKIP")
            continue

        raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"[{index:>3}/{len(images)}] {img_path.name} FAIL read")
            continue

        result = preprocess_palm_image(raw, config)
        roi_mask = roi_mask_from_result(result)
        compare = suppress_palm_lines(
            result["clahe"],
            roi_mask,
            final_size=config.final_size,
        )

        write_gray(baseline_dir / f"{stem}_final.png", result["final"])
        write_gray(compare_dir / f"{stem}_final.png", compare["final"])
        write_gray(vessel_dir / f"{stem}_thin_vein.png", compare["vessel_response"])
        visualize_comparison(viz_dir / f"{stem}_compare.png", result, compare)

        processed += 1
        print(f"[{index:>3}/{len(images)}] {img_path.name} OK")

    print("=" * 60)
    print(f"Finished. processed={processed}, skipped={skipped}")
    print(f"Compare outputs: {compare_dir}")


if __name__ == "__main__":
    main()

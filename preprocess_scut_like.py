"""SCUT-like batch preprocessing for raw palm captures.

This script mirrors the original SCUT preprocessing pipeline as closely as
possible:
- fixed 384x384 ROI centered by contour centroid + intensity-weighted centroid
- CLAHE with clip=2.0 and tile=8x8
- min-max normalization
- resize to 224x224 with Lanczos

It is intended for side-by-side comparison against the live-capture pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "captures/res_1920x1080_dataset_v3/raw"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "captures/res_1920x1080_dataset_v3/scut_like"
ROI_SIZE = 384
FINAL_SIZE = 224
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_palm_mask(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return mask


def palm_contour_center(mask: np.ndarray) -> tuple[int, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = mask.shape
        return (w // 2, h // 2)

    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        h, w = mask.shape
        return (w // 2, h // 2)
    return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))


def intensity_weighted_centroid(
    gray: np.ndarray,
    mask: np.ndarray,
    rough_center: tuple[int, int],
    window: int = 180,
) -> tuple[int, int]:
    cx, cy = rough_center
    h, w = gray.shape

    x1 = max(0, cx - window)
    x2 = min(w, cx + window)
    y1 = max(0, cy - window)
    y2 = min(h, cy + window)

    patch = gray[y1:y2, x1:x2].astype(np.float64)
    mask_patch = mask[y1:y2, x1:x2].astype(np.float64) / 255.0
    weighted = patch * mask_patch
    total = weighted.sum() + 1e-9

    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    refined_x = int((xs * weighted).sum() / total) + x1
    refined_y = int((ys * weighted).sum() / total) + y1
    return (refined_x, refined_y)


def extract_roi(gray: np.ndarray, roi_size: int = ROI_SIZE) -> tuple[np.ndarray, dict]:
    palm_mask = get_palm_mask(gray)
    rough_center = palm_contour_center(palm_mask)
    refined_center = intensity_weighted_centroid(gray, palm_mask, rough_center)

    cx, cy = refined_center
    half = roi_size // 2
    h, w = gray.shape

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    if x2 > w:
        x2 = w
        x1 = x2 - roi_size
    if y2 > h:
        y2 = h
        y1 = y2 - roi_size

    x1, y1 = max(0, x1), max(0, y1)
    roi = gray[y1:y2, x1:x2]

    if roi.shape[0] < roi_size or roi.shape[1] < roi_size:
        padded = np.zeros((roi_size, roi_size), dtype=np.uint8)
        padded[: roi.shape[0], : roi.shape[1]] = roi
        roi = padded

    debug = {
        "rough_center": rough_center,
        "refined_center": refined_center,
        "roi_box": (int(x1), int(y1), int(x2), int(y2)),
        "palm_mask": palm_mask,
    }
    return roi, debug


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return clahe.apply(gray)


def normalize_and_resize(gray: np.ndarray, final_size: int = FINAL_SIZE) -> np.ndarray:
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.resize(
        normalized,
        (final_size, final_size),
        interpolation=cv2.INTER_LANCZOS4,
    )


def preprocess_gray(gray: np.ndarray) -> dict:
    roi, debug = extract_roi(gray)
    clahe = apply_clahe(roi)
    final = normalize_and_resize(clahe)
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


def visualize(path: Path, stem: str, result: dict) -> None:
    raw = result["raw"]
    roi = result["roi"]
    clahe = result["clahe"]
    final = result["final"]
    debug = result["debug"]
    mask = debug["palm_mask"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"SCUT-like: {stem}", fontsize=12)

    ax = axes[0]
    ax.imshow(raw, cmap="gray")
    rc = debug["rough_center"]
    ref = debug["refined_center"]
    box = debug["roi_box"]
    ax.plot(*rc, "b+", markersize=12, markeredgewidth=2)
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
    axes[2].set_title("3. ROI 384x384")
    axes[2].axis("off")

    axes[3].imshow(clahe, cmap="gray")
    axes[3].set_title("4. CLAHE")
    axes[3].axis("off")

    axes[4].imshow(final, cmap="gray")
    axes[4].set_title("5. Final 224x224")
    axes[4].axis("off")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def image_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCUT-like preprocessing for raw capture data.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-skip", action="store_true", help="Reprocess existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = image_files(args.input_dir)
    if not images:
        raise SystemExit(f"No images found in: {args.input_dir}")

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
        final_path = final_dir / f"{stem}_final.png"
        if final_path.exists() and not args.no_skip:
            skipped += 1
            print(f"[{index:>3}/{len(images)}] {img_path.name} SKIP")
            continue

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[{index:>3}/{len(images)}] {img_path.name} FAIL read")
            continue

        result = preprocess_gray(gray)
        write_gray(final_path, result["final"])
        write_gray(roi_dir / f"{stem}_roi.png", result["roi"])
        write_gray(clahe_dir / f"{stem}_clahe.png", result["clahe"])
        visualize(viz_dir / f"{stem}_pipeline.png", stem, result)

        processed += 1
        print(f"[{index:>3}/{len(images)}] {img_path.name} OK")

    print("=" * 60)
    print(f"Finished. processed={processed}, skipped={skipped}")
    print(f"Final images: {final_dir}")


if __name__ == "__main__":
    main()

"""
Batch preprocessing for rejected 1920x1080 palm captures.

Input:
    captures/res_1920x1080_dataset_v3/rejected/raw

Output:
    captures/res_1920x1080_dataset_v3/rejected/result

Pipeline:
1. Segment palm/hand from dark background.
2. Estimate the palm core by ignoring narrow finger rows.
3. Extract an adaptive square ROI covering 90% of the detected palm-core span.
3. Apply CLAHE.
4. Normalize and resize to 224x224.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "captures/res_1920x1080_dataset_v3/raw"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "captures/res_1920x1080_dataset_v3/result"

ROI_SCALE = 0.90
PALM_CORE_WIDTH_RATIO = 0.60
FINAL_SIZE = 224
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
SKIP_DONE = True
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_metadata(img_path: Path) -> dict:
    """Read optional capture metadata stored next to the image."""
    meta_path = img_path.with_suffix(".json")
    if not meta_path.exists():
        return {}

    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def get_palm_mask(gray: np.ndarray) -> np.ndarray:
    """Segment the bright palm/hand from the dark capture background."""
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = gray.shape
    kernel_size = max(15, int(round(min(h, w) * 0.018)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return keep_largest_component(mask)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Drop small thresholding artifacts and keep only the main hand component."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    if not contours:
        return cleaned

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(cleaned, [largest], -1, 255, thickness=cv2.FILLED)
    return cleaned


def contour_stats(mask: np.ndarray, image_shape: tuple[int, int]) -> dict:
    """Return center and bounding box of the largest palm contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image_shape
    if not contours:
        return {
            "found": False,
            "area": 0.0,
            "bbox": (0, 0, w, h),
            "center": (w // 2, h // 2),
        }

    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    moments = cv2.moments(largest)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx = x + bw // 2
        cy = y + bh // 2

    return {
        "found": True,
        "area": float(cv2.contourArea(largest)),
        "bbox": (int(x), int(y), int(x + bw), int(y + bh)),
        "center": (int(cx), int(cy)),
    }


def largest_true_run(flags: np.ndarray) -> Optional[tuple[int, int]]:
    """Return [start, end) of the longest contiguous True run."""
    best_start = None
    best_len = 0
    current_start = None

    for idx, value in enumerate(flags):
        if value and current_start is None:
            current_start = idx
        if (not value or idx == len(flags) - 1) and current_start is not None:
            end = idx + 1 if value and idx == len(flags) - 1 else idx
            run_len = end - current_start
            if run_len > best_len:
                best_start = current_start
                best_len = run_len
            current_start = None

    if best_start is None:
        return None
    return (int(best_start), int(best_start + best_len))


def palm_core_bbox(
    mask: np.ndarray,
    fallback_bbox: tuple[int, int, int, int],
    width_ratio: float = PALM_CORE_WIDTH_RATIO,
) -> tuple[int, int, int, int]:
    """
    Estimate palm-only bbox by keeping rows where the hand mask is wide.

    Finger rows are narrow or split by gaps, while palm rows have a broad
    continuous mask. The largest wide-row band gives a practical palm core for
    ROI extraction on full-hand captures.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return fallback_bbox

    row_widths = np.zeros(mask.shape[0], dtype=np.int32)
    for y in np.unique(ys):
        x_values = xs[ys == y]
        row_widths[y] = int(x_values.max() - x_values.min() + 1)

    max_width = int(row_widths.max())
    if max_width <= 0:
        return fallback_bbox

    wide_rows = row_widths >= int(round(max_width * width_ratio))
    run = largest_true_run(wide_rows)
    if run is None:
        return fallback_bbox

    y1, y2 = run
    min_height = max(1, int((fallback_bbox[3] - fallback_bbox[1]) * 0.25))
    if y2 - y1 < min_height:
        return fallback_bbox

    core = mask[y1:y2, :]
    core_ys, core_xs = np.where(core > 0)
    if len(core_xs) == 0:
        return fallback_bbox

    x1 = int(core_xs.min())
    x2 = int(core_xs.max() + 1)
    return (x1, int(y1), x2, int(y2))


def intensity_weighted_centroid(
    gray: np.ndarray,
    mask: np.ndarray,
    rough_center: tuple[int, int],
    window: int,
) -> tuple[int, int]:
    """Shift the ROI center toward brighter palm pixels inside the mask."""
    cx, cy = rough_center
    h, w = gray.shape
    x1 = max(0, cx - window)
    x2 = min(w, cx + window)
    y1 = max(0, cy - window)
    y2 = min(h, cy + window)

    patch = gray[y1:y2, x1:x2].astype(np.float64)
    mask_patch = mask[y1:y2, x1:x2].astype(np.float64) / 255.0
    weighted = patch * mask_patch
    total = weighted.sum()
    if total <= 0:
        return rough_center

    ys, xs = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]
    refined_x = int((xs * weighted).sum() / total) + x1
    refined_y = int((ys * weighted).sum() / total) + y1
    return (refined_x, refined_y)


def clamp_square_box(
    center: tuple[int, int],
    side: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Build a square crop box and keep it inside image bounds."""
    h, w = image_shape
    side = int(min(max(1, side), w, h))
    half = side // 2
    cx, cy = center

    x1 = int(cx - half)
    y1 = int(cy - half)
    x2 = x1 + side
    y2 = y1 + side

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= x2 - w
        x2 = w
    if y2 > h:
        y1 -= y2 - h
        y2 = h

    return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))


def extract_adaptive_roi(
    gray: np.ndarray,
    roi_scale: float = ROI_SCALE,
) -> tuple[np.ndarray, dict]:
    """
    Extract ROI covering roi_scale of the detected palm-core span.

    For 1920x1080 captures the hand often touches the top/bottom border, so a
    fixed 384px ROI is too small. This estimates a palm-core bbox from the
    largest contour, takes 90% of the smaller palm-core dimension as a square
    ROI side, then centers it using an intensity-weighted centroid.
    """
    if not (0.0 < roi_scale <= 1.0):
        raise ValueError("roi_scale must satisfy 0 < roi_scale <= 1")

    palm_mask = get_palm_mask(gray)
    stats = contour_stats(palm_mask, gray.shape)
    core_bbox = palm_core_bbox(palm_mask, stats["bbox"])
    x1, y1, x2, y2 = core_bbox
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    side = int(round(min(bbox_w, bbox_h) * roi_scale))

    window = max(side // 2, 120)
    core_center = (x1 + bbox_w // 2, y1 + bbox_h // 2)
    refined_center = intensity_weighted_centroid(
        gray,
        palm_mask,
        core_center,
        window=window,
    )
    roi_box = clamp_square_box(refined_center, side, gray.shape)
    rx1, ry1, rx2, ry2 = roi_box
    roi = gray[ry1:ry2, rx1:rx2]

    debug = {
        "palm_mask": palm_mask,
        "hand_bbox": stats["bbox"],
        "palm_bbox": core_bbox,
        "palm_area": stats["area"],
        "rough_center": core_center,
        "hand_center": stats["center"],
        "refined_center": refined_center,
        "roi_box": roi_box,
        "roi_scale": float(roi_scale),
        "roi_side": int(side),
        "centroid_window": int(window),
    }
    return roi, debug


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = CLAHE_CLIP,
    tile_grid: tuple[int, int] = CLAHE_TILE,
) -> np.ndarray:
    """Apply CLAHE to enhance local palm-vein contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def normalize_and_resize(gray: np.ndarray, final_size: int = FINAL_SIZE) -> np.ndarray:
    """Normalize intensity to [0, 255] and resize to model input size."""
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.resize(
        normalized,
        (final_size, final_size),
        interpolation=cv2.INTER_AREA,
    )


def preprocess_image(img_path: Path, roi_scale: float, final_size: int) -> dict:
    """Run full preprocessing pipeline on one image."""
    raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Cannot read image: {img_path}")

    roi, debug = extract_adaptive_roi(raw, roi_scale=roi_scale)
    clahe = apply_clahe(roi)
    final = normalize_and_resize(clahe, final_size=final_size)
    metadata = read_metadata(img_path)

    if metadata:
        debug["metadata_bbox"] = metadata.get("bbox")
        debug["metadata_center"] = metadata.get("center")

    return {
        "raw": raw,
        "roi": roi,
        "clahe": clahe,
        "final": final,
        "debug": debug,
        "path": img_path,
    }


def write_gray(path: Path, image: np.ndarray) -> None:
    """Write a grayscale image and fail loudly if OpenCV cannot encode it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"Failed to write image: {path}")


def visualize_pipeline(result: dict, save_path: Path) -> None:
    """Save a compact debug visualization for ROI tuning."""
    raw = result["raw"]
    roi = result["roi"]
    clahe = result["clahe"]
    final = result["final"]
    debug = result["debug"]
    mask = debug["palm_mask"]
    fname = result["path"].name

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"Adaptive 90% ROI preprocessing: {fname}", fontsize=13)

    ax = axes[0]
    ax.imshow(raw, cmap="gray")
    bbox = debug["palm_bbox"]
    roi_box = debug["roi_box"]
    rough_center = debug["rough_center"]
    refined_center = debug["refined_center"]
    ax.add_patch(
        plt.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
            label="Palm bbox",
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (roi_box[0], roi_box[1]),
            roi_box[2] - roi_box[0],
            roi_box[3] - roi_box[1],
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            label="90% ROI",
        )
    )
    ax.plot(*rough_center, "b+", markersize=14, markeredgewidth=2)
    ax.plot(*refined_center, "r+", markersize=14, markeredgewidth=2)
    ax.set_title(f"1. Original\n{raw.shape[1]}x{raw.shape[0]}")
    ax.axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("2. Palm Mask")
    axes[1].axis("off")

    axes[2].imshow(roi, cmap="gray")
    axes[2].set_title(f"3. ROI\n{roi.shape[1]}x{roi.shape[0]}")
    axes[2].axis("off")

    axes[3].imshow(clahe, cmap="gray")
    axes[3].set_title(f"4. CLAHE\nclip={CLAHE_CLIP}, tile={CLAHE_TILE}")
    axes[3].axis("off")

    axes[4].imshow(final, cmap="gray")
    axes[4].set_title(f"5. Final\n{FINAL_SIZE}x{FINAL_SIZE}")
    axes[4].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def image_files(input_dir: Path) -> list[Path]:
    """Return supported image files sorted by name."""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess rejected palm captures using adaptive 90% ROI.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--roi-scale", type=float, default=ROI_SCALE)
    parser.add_argument("--final-size", type=int, default=FINAL_SIZE)
    parser.add_argument("--no-skip", action="store_true", help="Reprocess existing outputs.")
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Only save final images, without ROI/mask/CLAHE/visualization files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    final_dir = output_dir / "final"
    roi_dir = output_dir / "roi"
    clahe_dir = output_dir / "clahe"
    mask_dir = output_dir / "mask"
    viz_dir = output_dir / "visualizations"

    images = image_files(input_dir)
    if not images:
        raise SystemExit(f"No images found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0
    print(f"Input    : {input_dir}")
    print(f"Output   : {output_dir}")
    print(f"Images   : {len(images)}")
    print(f"ROI scale: {args.roi_scale:.2f}")
    print("-" * 60)

    for idx, img_path in enumerate(images, start=1):
        stem = img_path.stem
        final_path = final_dir / f"{stem}_final.png"
        if SKIP_DONE and not args.no_skip and final_path.exists():
            skipped += 1
            print(f"[{idx:>3}/{len(images)}] {img_path.name} SKIP")
            continue

        try:
            result = preprocess_image(
                img_path,
                roi_scale=float(args.roi_scale),
                final_size=int(args.final_size),
            )
            write_gray(final_path, result["final"])

            if not args.no_debug:
                write_gray(roi_dir / f"{stem}_roi.png", result["roi"])
                write_gray(clahe_dir / f"{stem}_clahe.png", result["clahe"])
                write_gray(mask_dir / f"{stem}_mask.png", result["debug"]["palm_mask"])
                visualize_pipeline(result, viz_dir / f"{stem}_pipeline.png")

            processed += 1
            roi_box = result["debug"]["roi_box"]
            print(
                f"[{idx:>3}/{len(images)}] {img_path.name} OK "
                f"roi={roi_box} side={result['debug']['roi_side']}"
            )
        except Exception as exc:
            failed += 1
            print(f"[{idx:>3}/{len(images)}] {img_path.name} FAIL: {exc}")

    print("=" * 60)
    print(f"Finished. processed={processed}, skipped={skipped}, failed={failed}")
    print(f"Final images: {final_dir}")


if __name__ == "__main__":
    main()

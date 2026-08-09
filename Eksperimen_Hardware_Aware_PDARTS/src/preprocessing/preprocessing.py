"""
Palm Vein Preprocessing Pipeline
- Step 1: ROI Extraction using Gradient-Based Palm Center Detection + Intensity-Weighted Centroid
- Step 2: CLAHE for contrast enhancement
- Step 3: Intensity normalization + resize to 224x224
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt


# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR  = Path("C:\\Users\\Nanik Suciati\\Downloads\\Palm Vein Tesis\\SCUT_PV_V1_raw10")
OUTPUT_DIR   = Path("C:\\Users\\Nanik Suciati\\Downloads\\Palm Vein Tesis\\preprocessed_results")
ROI_SIZE     = 384          # ROI square size before final resize
FINAL_SIZE   = 224          # final output size
CLAHE_CLIP   = 2.0          # CLAHE clip limit
CLAHE_TILE   = (8, 8)       # CLAHE tile grid size
SKIP_DONE    = True         # skip subject if all images already processed


# ─── Step 1: ROI Extraction ────────────────────────────────────────────────────

def get_palm_mask(gray: np.ndarray) -> np.ndarray:
    """
    Segment palm from background via Otsu thresholding.
    In NIR palm-vein images: palm is bright, background is dark.
    Returns binary mask (255 = palm, 0 = background).
    """
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup: fill holes, remove tiny blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    return mask


def palm_contour_center(mask: np.ndarray) -> tuple:
    """
    Find centroid of the largest contour in the palm mask.
    Gives the rough geometric center of the palm.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = mask.shape
        return (w // 2, h // 2)  # fallback

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        h, w = mask.shape
        return (w // 2, h // 2)

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def intensity_weighted_centroid(gray: np.ndarray,
                                mask: np.ndarray,
                                rough_center: tuple,
                                window: int = 180) -> tuple:
    """
    Refine center using Intensity-Weighted Centroid INSIDE the palm mask.
    Brighter palm pixels carry more weight → shifts toward vein-rich region.
    """
    cx, cy = rough_center
    h, w = gray.shape

    x1 = max(0, cx - window)
    x2 = min(w, cx + window)
    y1 = max(0, cy - window)
    y2 = min(h, cy + window)

    patch      = gray[y1:y2, x1:x2].astype(np.float64)
    mask_patch = mask[y1:y2, x1:x2].astype(np.float64) / 255.0

    # Weight = pixel intensity × within-palm (ignore background)
    weighted = patch * mask_patch
    total    = weighted.sum() + 1e-9

    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    refined_x = int((xs * weighted).sum() / total) + x1
    refined_y = int((ys * weighted).sum() / total) + y1

    return (refined_x, refined_y)


def extract_roi(gray: np.ndarray, roi_size: int = ROI_SIZE) -> tuple:
    """
    Extract square ROI centered on the detected palm center.
    1. Otsu mask to find palm region
    2. Largest contour centroid = rough center
    3. Intensity-Weighted Centroid (within mask) = refined center
    Returns cropped ROI and debug info dict.
    """
    palm_mask      = get_palm_mask(gray)
    rough_center   = palm_contour_center(palm_mask)
    refined_center = intensity_weighted_centroid(gray, palm_mask, rough_center)

    cx, cy = refined_center
    half   = roi_size // 2
    h, w   = gray.shape

    # Clamp so ROI stays within image bounds
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

    # Pad if image is smaller than roi_size (edge case)
    if roi.shape[0] < roi_size or roi.shape[1] < roi_size:
        padded = np.zeros((roi_size, roi_size), dtype=np.uint8)
        padded[:roi.shape[0], :roi.shape[1]] = roi
        roi = padded

    debug = {
        "rough_center":   rough_center,
        "refined_center": refined_center,
        "roi_box":        (x1, y1, x2, y2),
        "palm_mask":      palm_mask,
    }
    return roi, debug


# ─── Step 2: CLAHE ────────────────────────────────────────────────────────────

def apply_clahe(gray: np.ndarray,
                clip_limit: float = CLAHE_CLIP,
                tile_grid: tuple  = CLAHE_TILE) -> np.ndarray:
    """Apply CLAHE to enhance vein contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


# ─── Step 3: Normalize + Resize ───────────────────────────────────────────────

def normalize_and_resize(gray: np.ndarray,
                         final_size: int = FINAL_SIZE) -> np.ndarray:
    """
    Normalize intensity to [0, 255] then resize to final_size × final_size.
    """
    # Intensity normalization (min-max)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Resize with Lanczos (high quality downsampling)
    resized = cv2.resize(normalized, (final_size, final_size),
                         interpolation=cv2.INTER_LANCZOS4)
    return resized


# ─── Full Pipeline ─────────────────────────────────────────────────────────────

def preprocess_image(img_path: Path) -> dict:
    """Run full preprocessing pipeline on a single image."""
    raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Cannot read image: {img_path}")

    roi,   debug  = extract_roi(raw)
    clahe_img     = apply_clahe(roi)
    final_img     = normalize_and_resize(clahe_img)

    return {
        "raw":    raw,
        "roi":    roi,
        "clahe":  clahe_img,
        "final":  final_img,
        "debug":  debug,
        "path":   img_path,
    }


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_pipeline(result: dict, save_path: Optional[Path] = None):
    """Show 5-stage pipeline: Original → Palm Mask → ROI → CLAHE → Final."""
    raw   = result["raw"]
    roi   = result["roi"]
    clahe = result["clahe"]
    final = result["final"]
    debug = result["debug"]
    fname = result["path"].name
    mask  = debug["palm_mask"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"Pipeline: {fname}", fontsize=13, fontweight="bold")

    # Stage 1 — Original with centers and ROI box
    ax = axes[0]
    ax.imshow(raw, cmap="gray")
    rc  = debug["rough_center"]
    ref = debug["refined_center"]
    box = debug["roi_box"]
    ax.plot(*rc,  "b+", markersize=14, markeredgewidth=2, label="Contour center")
    ax.plot(*ref, "r+", markersize=14, markeredgewidth=2, label="Refined center")
    rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                          linewidth=2, edgecolor="lime", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(f"1. Original\n{raw.shape[1]}×{raw.shape[0]}", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.axis("off")

    # Stage 2 — Palm Mask (Otsu)
    ax = axes[1]
    overlay = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    overlay[mask == 0] = [0, 0, 0]  # black out background
    ax.imshow(overlay)
    ax.set_title("2. Palm Mask\n(Otsu)", fontsize=10)
    ax.axis("off")

    # Stage 3 — ROI
    ax = axes[2]
    ax.imshow(roi, cmap="gray")
    ax.set_title(f"3. ROI Crop\n{ROI_SIZE}×{ROI_SIZE} px", fontsize=10)
    ax.axis("off")

    # Stage 4 — CLAHE
    ax = axes[3]
    ax.imshow(clahe, cmap="gray")
    ax.set_title(f"4. CLAHE\nclip={CLAHE_CLIP}, tile={CLAHE_TILE}", fontsize=10)
    ax.axis("off")

    # Stage 5 — Final
    ax = axes[4]
    ax.imshow(final, cmap="gray")
    ax.set_title(f"5. Normalized + Resize\n{FINAL_SIZE}×{FINAL_SIZE} px", fontsize=10)
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_subject_strip(subject_id: str, results: list, save_path: Optional[Path] = None):
    """Strip showing all 10 final outputs for one subject side-by-side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    fig.suptitle(f"Subject {subject_id} — Final Outputs ({FINAL_SIZE}×{FINAL_SIZE})",
                 fontsize=12, fontweight="bold")
    for i, (ax, res) in enumerate(zip(axes, results)):
        ax.imshow(res["final"], cmap="gray")
        ax.set_title(res["path"].stem, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def get_all_subjects() -> list:
    """Return sorted list of all numeric subject folder names."""
    subjects = [
        d.name for d in DATASET_DIR.iterdir()
        if d.is_dir() and d.name.isdigit()
    ]
    return sorted(subjects, key=lambda x: int(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess SCUT_PV_v1 into the 224x224 representation used by the thesis."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip per-image and per-subject diagnostic figures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess images that already exist in the output directory.",
    )
    return parser.parse_args()


def main():
    global DATASET_DIR, OUTPUT_DIR, SKIP_DONE
    args = parse_args()
    DATASET_DIR = args.input_dir.resolve()
    OUTPUT_DIR = args.output_dir.resolve()
    SKIP_DONE = not args.overwrite

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_subjects = get_all_subjects()
    total_subjects = len(all_subjects)
    total_images   = sum(
        len(list((DATASET_DIR / s).glob("*.bmp")))
        for s in all_subjects
    )

    print(f"Dataset  : {DATASET_DIR}")
    print(f"Subjects : {total_subjects}")
    print(f"Images   : {total_images}")
    print(f"Output   : {OUTPUT_DIR}")
    print("-" * 60)

    processed_total = 0
    skipped_total   = 0

    for subj_idx, subj_id in enumerate(all_subjects, 1):
        subj_dir = DATASET_DIR / subj_id
        subj_out = OUTPUT_DIR  / subj_id
        viz_dir  = OUTPUT_DIR  / "visualizations" / subj_id

        images = sorted(subj_dir.glob("*.bmp"),
                        key=lambda p: int(p.stem.split("_")[1]))

        # --- Skip if all output images already exist ---
        if SKIP_DONE:
            already_done = all((subj_out / img.name).exists() for img in images)
            if already_done and len(images) > 0:
                skipped_total += len(images)
                print(f"[{subj_idx:>4}/{total_subjects}] Subject {subj_id:>4}  SKIP (already processed)")
                continue

        subj_out.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)

        subject_results = []
        print(f"[{subj_idx:>4}/{total_subjects}] Subject {subj_id:>4}  ({len(images)} images)")

        for img_path in images:
            out_path = subj_out / img_path.name

            # skip single image if done
            if SKIP_DONE and out_path.exists():
                skipped_total += 1
                continue

            result = preprocess_image(img_path)
            subject_results.append(result)

            # Save processed image
            cv2.imwrite(str(out_path), result["final"])
            processed_total += 1

            # Save per-image pipeline visualization
            if not args.no_visualizations:
                viz_path = viz_dir / f"{img_path.stem}_pipeline.png"
                visualize_pipeline(result, save_path=viz_path)

        # Save subject strip (only if we processed at least one new image)
        if subject_results and not args.no_visualizations:
            strip_path = viz_dir / f"subject_{subj_id}_strip.png"
            visualize_subject_strip(subj_id, subject_results, save_path=strip_path)

        # Progress summary line
        pct = (subj_idx / total_subjects) * 100
        print(f"           └─ done  |  total processed: {processed_total}  skipped: {skipped_total}  [{pct:.1f}%]")

    print(f"\n{'='*60}")
    print(f"✅ Finished!")
    print(f"   Processed : {processed_total} images")
    print(f"   Skipped   : {skipped_total} images (already existed)")
    print(f"   Output    : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

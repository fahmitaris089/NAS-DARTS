"""Adaptive preprocessing for the 2 new palm-vein classes (835/836).

This script preprocesses raw captures stored in:
    captures/final_dataset/left
    captures/final_dataset/right

Output is written to:
    captures/final_dataset/preprocessed/835
    captures/final_dataset/preprocessed/836

Pipeline policy matches the teacher-like downstream stages:
- adaptive palm-core ROI
- CLAHE
- min-max normalization
- resize to 224x224

Raw input files are never modified.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from palm_preprocessing import (
    PROFILE_DATASET_V3,
    PalmPreprocessingConfig,
    apply_clahe,
    assess_palm_vein_quality,
    clamp_square_box,
    contour_stats,
    create_vessel_preview,
    get_palm_mask,
    intensity_weighted_centroid,
    palm_core_bbox,
    resolve_centroid_window,
    resize_final,
    summarize_image_quality,
)


ROOT_DIR = Path(__file__).resolve().parent
INPUT_ROOT = ROOT_DIR / "captures" / "final_dataset"
OUTPUT_ROOT = INPUT_ROOT / "preprocessed"
RELAXED_PALM_WIDTH_RATIO = 0.60
RELAXED_CORE_WIDTH_WEIGHT = 0.60
RELAXED_CORE_HEIGHT_WEIGHT = 1.35
RELAXED_HAND_HEIGHT_WEIGHT = 0.72
RELAXED_MIN_SIDE = 560
RELAXED_CENTER_Y_SHIFT = 0.10
RELAXED_MIN_CORE_HEIGHT_RATIO = 0.60
RELAXED_MAX_CORE_TOP_OFFSET_RATIO = 0.18

CLASS_MAP = {
    "left": "835",
    "right": "836",
}

CONFIG = PalmPreprocessingConfig(
    roi_size=384,
    final_size=224,
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    centroid_window=0,
    profile=PROFILE_DATASET_V3,
    denoise_h=0.0,
    vessel_preview_kernel=17,
    center_offset_x=0,
    center_offset_y=0,
    stretch_percentiles=None,
    adaptive_roi=True,
    adaptive_roi_scale=0.90,
    palm_core_width_ratio=0.60,
)


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )


def read_raw_metadata(image_path: Path) -> dict[str, Any] | None:
    meta_path = image_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def extract_relaxed_adaptive_roi(gray: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract a looser ROI so most of the palm remains visible after resize."""
    palm_mask = get_palm_mask(gray)
    stats = contour_stats(palm_mask, gray.shape)
    candidate_core_bbox = palm_core_bbox(
        palm_mask,
        stats["bbox"],
        RELAXED_PALM_WIDTH_RATIO,
    )
    hand_x1, hand_y1, hand_x2, hand_y2 = stats["bbox"]
    hand_height = max(1, hand_y2 - hand_y1)

    core_x1, core_y1, core_x2, core_y2 = candidate_core_bbox
    core_height = max(1, core_y2 - core_y1)
    core_top_offset = max(0, core_y1 - hand_y1)
    core_looks_collapsed = (
        core_height < int(round(hand_height * RELAXED_MIN_CORE_HEIGHT_RATIO))
        or core_top_offset > int(round(hand_height * RELAXED_MAX_CORE_TOP_OFFSET_RATIO))
    )
    core_bbox = stats["bbox"] if core_looks_collapsed else candidate_core_bbox

    x1, y1, x2, y2 = core_bbox
    core_width = max(1, x2 - x1)
    core_height = max(1, y2 - y1)
    hand_height = max(1, hand_y2 - hand_y1)

    side = int(
        round(
            max(
                core_width * RELAXED_CORE_WIDTH_WEIGHT,
                core_height * RELAXED_CORE_HEIGHT_WEIGHT,
                hand_height * RELAXED_HAND_HEIGHT_WEIGHT,
                RELAXED_MIN_SIDE,
            )
        )
    )

    resolved_window = resolve_centroid_window(gray.shape, side, CONFIG.centroid_window)
    rough_center = (x1 + core_width // 2, y1 + core_height // 2)
    weighted_center = intensity_weighted_centroid(
        gray,
        palm_mask,
        rough_center,
        resolved_window,
    )
    shift_y = int(round(core_height * RELAXED_CENTER_Y_SHIFT))
    refined_center = (
        weighted_center[0],
        weighted_center[1] + shift_y,
    )
    roi_box = clamp_square_box(refined_center, side, gray.shape)
    rx1, ry1, rx2, ry2 = roi_box
    roi = gray[ry1:ry2, rx1:rx2]

    debug = {
        "rough_center": rough_center,
        "weighted_center": weighted_center,
        "center_before_offset": weighted_center,
        "center_after_offset": refined_center,
        "refined_center": refined_center,
        "center_offset": (0, shift_y),
        "center_mode": "adaptive_weighted_relaxed",
        "roi_box": roi_box,
        "palm_mask": palm_mask,
        "centroid_window": int(resolved_window),
        "adaptive_roi": True,
        "adaptive_roi_scale": float(CONFIG.adaptive_roi_scale),
        "palm_core_width_ratio": float(RELAXED_PALM_WIDTH_RATIO),
        "candidate_palm_bbox": candidate_core_bbox,
        "core_bbox_fallback_applied": bool(core_looks_collapsed),
        "core_bbox_fallback_reason": {
            "core_height": int(core_y2 - core_y1),
            "hand_height": int(hand_height),
            "core_top_offset": int(core_top_offset),
            "min_core_height": int(round(hand_height * RELAXED_MIN_CORE_HEIGHT_RATIO)),
            "max_core_top_offset": int(round(hand_height * RELAXED_MAX_CORE_TOP_OFFSET_RATIO)),
        },
        "hand_bbox": stats["bbox"],
        "hand_center": stats["center"],
        "palm_bbox": core_bbox,
        "palm_area": float(stats["area"]),
        "roi_side": int(side),
        "roi_strategy": {
            "core_width_weight": RELAXED_CORE_WIDTH_WEIGHT,
            "core_height_weight": RELAXED_CORE_HEIGHT_WEIGHT,
            "hand_height_weight": RELAXED_HAND_HEIGHT_WEIGHT,
            "min_side": RELAXED_MIN_SIDE,
            "center_y_shift_ratio": RELAXED_CENTER_Y_SHIFT,
            "min_core_height_ratio": RELAXED_MIN_CORE_HEIGHT_RATIO,
            "max_core_top_offset_ratio": RELAXED_MAX_CORE_TOP_OFFSET_RATIO,
        },
    }
    return roi, debug


def preprocess_gray(gray: np.ndarray) -> dict[str, Any]:
    """Teacher-like post-ROI pipeline with a looser adaptive ROI for device captures."""
    raw = gray.astype(np.uint8, copy=True)
    roi, debug = extract_relaxed_adaptive_roi(raw)
    clahe = apply_clahe(roi, CONFIG.clahe_clip, CONFIG.clahe_tile)
    final_source = cv2.normalize(clahe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    final = resize_final(final_source, CONFIG.final_size, interpolation=cv2.INTER_AREA)
    vessel_preview = create_vessel_preview(clahe, CONFIG.vessel_preview_kernel)
    quality_filter = assess_palm_vein_quality(final)

    debug.update(
        {
            "profile": CONFIG.profile,
            "roi_size": int(CONFIG.roi_size),
            "final_size": int(CONFIG.final_size),
            "clahe_clip": float(CONFIG.clahe_clip),
            "clahe_tile": [int(CONFIG.clahe_tile[0]), int(CONFIG.clahe_tile[1])],
            "denoise_h": float(CONFIG.denoise_h),
            "vessel_preview_kernel": int(CONFIG.vessel_preview_kernel),
            "resize_interpolation": int(cv2.INTER_AREA),
            "stretch_percentiles": None,
            "quality": {
                "raw": summarize_image_quality(raw),
                "roi": summarize_image_quality(roi),
                "clahe": summarize_image_quality(clahe),
                "final_source": summarize_image_quality(final_source),
                "final": summarize_image_quality(final),
            },
            "quality_filter": quality_filter,
        }
    )
    return {
        "raw": raw,
        "roi": roi,
        "clahe": clahe,
        "final": final,
        "vessel_preview": vessel_preview,
        "debug": debug,
    }


def preprocess_debug_payload(
    image_path: Path,
    class_id: str,
    raw_metadata: dict[str, Any] | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    debug = result["debug"]
    raw_meta_path = image_path.with_suffix(".json")
    payload = {
        "source_raw_image": str(image_path),
        "source_raw_metadata": str(raw_meta_path) if raw_meta_path.exists() else None,
        "class_id": class_id,
        "preprocessing_config": asdict(CONFIG),
        "roi_box": list(debug.get("roi_box", ())),
        "rough_center": list(debug.get("rough_center", ())),
        "weighted_center": list(debug.get("weighted_center", ())),
        "final_center": list(
            debug.get(
                "center_after_offset",
                debug.get("refined_center", ()),
            )
        ),
        "hand_bbox": list(debug.get("hand_bbox", ())),
        "palm_bbox": list(debug.get("palm_bbox", ())),
        "quality": debug.get("quality"),
        "quality_filter": debug.get("quality_filter"),
        "raw_capture_metadata": raw_metadata,
    }
    return payload


def process_folder(side_name: str, class_id: str) -> tuple[int, int]:
    input_dir = INPUT_ROOT / side_name
    output_dir = OUTPUT_ROOT / class_id
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for image_path in image_files(input_dir):
        stem = image_path.stem
        png_out = output_dir / f"{stem}.png"
        bmp_out = output_dir / f"{stem}.bmp"
        json_out = output_dir / f"{stem}_preprocess.json"

        if png_out.exists() and bmp_out.exists() and json_out.exists():
            skipped += 1
            continue

        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Cannot read image: {image_path}")

        raw_metadata = read_raw_metadata(image_path)
        result = preprocess_gray(gray)
        final = result["final"]

        if not cv2.imwrite(str(png_out), final):
            raise OSError(f"Failed to write {png_out}")
        if not cv2.imwrite(str(bmp_out), final):
            raise OSError(f"Failed to write {bmp_out}")

        debug_payload = preprocess_debug_payload(
            image_path=image_path,
            class_id=class_id,
            raw_metadata=raw_metadata,
            result=result,
        )
        json_out.write_text(
            json.dumps(debug_payload, indent=2),
            encoding="utf-8",
        )
        processed += 1

    return processed, skipped


def main() -> None:
    print(f"Input root : {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Config     : {CONFIG}")
    print("-" * 72)

    total_processed = 0
    total_skipped = 0

    for side_name, class_id in CLASS_MAP.items():
        input_dir = INPUT_ROOT / side_name
        output_dir = OUTPUT_ROOT / class_id
        images = image_files(input_dir)
        print(
            f"{side_name:>5} -> class {class_id}: "
            f"{len(images)} input images | output {output_dir}"
        )
        processed, skipped = process_folder(side_name, class_id)
        total_processed += processed
        total_skipped += skipped
        print(
            f"        processed={processed} skipped={skipped} "
            f"output={output_dir}"
        )

    print("-" * 72)
    print(
        f"Done. processed={total_processed} skipped={total_skipped} "
        f"output_root={OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()

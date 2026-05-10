"""Batch preprocess sharper 1920x1080 palm captures with a softer dataset_v3 variant."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from palm_preprocessing import (
    PROFILE_DATASET_V3_SOFT,
    PalmPreprocessingConfig,
    preprocess_palm_image,
)


INPUT_DIR = Path(
    "/Users/fahmitaris/Downloads/NAS-DARTS/captures/res_1920x1080_dataset_v3_try2/raw"
)
OUTPUT_DIR = Path(
    "/Users/fahmitaris/Downloads/NAS-DARTS/captures/res_1920x1080_dataset_v3_try2/dataset_v3_soft"
)

CONFIG = PalmPreprocessingConfig(
    profile=PROFILE_DATASET_V3_SOFT,
    roi_size=760,
    final_size=224,
    clahe_clip=1.6,
    clahe_tile=(12, 12),
    denoise_h=2.0,
    adaptive_roi=True,
    adaptive_roi_scale=0.95,
    palm_core_width_ratio=0.45,
    stretch_percentiles=None,
)


def write_gray(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    raw_images = sorted(INPUT_DIR.glob("*.png"))
    if not raw_images:
        raise SystemExit(f"No PNG files found in {INPUT_DIR}")

    final_dir = OUTPUT_DIR / "final"
    processed_dir = OUTPUT_DIR / "processed"
    vis_dir = OUTPUT_DIR / "visualizations"
    meta_dir = OUTPUT_DIR / "metadata"
    final_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for raw_path in raw_images:
        gray = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"skip unreadable: {raw_path.name}")
            continue

        result = preprocess_palm_image(gray, CONFIG)
        stem = raw_path.stem

        write_gray(final_dir / f"{stem}_final.png", result["final"])
        write_gray(processed_dir / f"{stem}_roi.png", result["roi"])
        write_gray(processed_dir / f"{stem}_mask.png", result["mask"])
        write_gray(processed_dir / f"{stem}_clahe.png", result["clahe"])
        write_gray(vis_dir / f"{stem}_vessel_preview.png", result["vessel_preview"])

        raw_json_path = raw_path.with_suffix(".json")
        raw_metadata = {}
        if raw_json_path.exists():
            raw_metadata = json.loads(raw_json_path.read_text())

        debug = to_jsonable(result["debug"])
        metadata = {
            "source_image": raw_path.name,
            "source_json": raw_json_path.name if raw_json_path.exists() else None,
            "config": {
                "profile": CONFIG.profile,
                "roi_size": CONFIG.roi_size,
                "final_size": CONFIG.final_size,
                "clahe_clip": CONFIG.clahe_clip,
                "clahe_tile": list(CONFIG.clahe_tile),
                "denoise_h": CONFIG.denoise_h,
                "adaptive_roi": CONFIG.adaptive_roi,
                "adaptive_roi_scale": CONFIG.adaptive_roi_scale,
                "palm_core_width_ratio": CONFIG.palm_core_width_ratio,
                "stretch_percentiles": CONFIG.stretch_percentiles,
            },
            "preprocessing_debug": debug,
            "source_camera_settings": raw_metadata.get("camera_settings"),
            "source_quality_filter": raw_metadata.get("quality_filter"),
        }
        (meta_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2))
        processed_count += 1

    print(f"Input     : {INPUT_DIR}")
    print(f"Output    : {OUTPUT_DIR}")
    print(f"Processed : {processed_count} images")


if __name__ == "__main__":
    main()

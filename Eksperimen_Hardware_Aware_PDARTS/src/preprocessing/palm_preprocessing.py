"""Palm preprocessing helpers for live capture and batch workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


PROFILE_LEGACY = "legacy"
PROFILE_CAPTURE_V2 = "capture_v2"
PROFILE_DATASET_V3 = "dataset_v3"
PROFILE_DATASET_V3_SOFT = "dataset_v3_soft"
SUPPORTED_PROFILES = (
    PROFILE_LEGACY,
    PROFILE_CAPTURE_V2,
    PROFILE_DATASET_V3,
    PROFILE_DATASET_V3_SOFT,
)


@dataclass(frozen=True)
class PalmQualityFilterConfig:
    min_final_mean: float = 110.0
    min_final_p95: float = 125.0
    min_final_std: float = 16.0
    max_dark_fraction_lt_110: float = 0.85
    min_laplacian_var: float = 18.0
    min_gradient_p95: float = 40.0
    min_texture_to_gradient_ratio: float = 0.55
    bright_threshold: int = 220
    dark_threshold: int = 110


@dataclass(frozen=True)
class PalmPreprocessingConfig:
    roi_size: int = 384
    final_size: int = 224
    clahe_clip: float = 2.0
    clahe_tile: Tuple[int, int] = (8, 8)
    centroid_window: int = 0
    profile: str = PROFILE_LEGACY
    denoise_h: float = 0.0
    vessel_preview_kernel: int = 17
    center_offset_x: int = 0
    center_offset_y: int = 0
    stretch_percentiles: Tuple[float, float] | None = None
    adaptive_roi: bool = False
    adaptive_roi_scale: float = 0.9
    palm_core_width_ratio: float = 0.6


def validate_preprocessing_config(config: PalmPreprocessingConfig) -> None:
    if config.profile not in SUPPORTED_PROFILES:
        raise ValueError(
            "profile must be one of: " + ", ".join(SUPPORTED_PROFILES)
        )
    if config.roi_size <= 0:
        raise ValueError("roi_size must be positive")
    if config.final_size <= 0:
        raise ValueError("final_size must be positive")
    if config.clahe_clip <= 0.0:
        raise ValueError("clahe_clip must be positive")
    if config.clahe_tile[0] <= 0 or config.clahe_tile[1] <= 0:
        raise ValueError("clahe_tile dimensions must be positive")
    if config.centroid_window < 0:
        raise ValueError("centroid_window must be >= 0")
    if config.denoise_h < 0.0:
        raise ValueError("denoise_h must be >= 0")
    if config.vessel_preview_kernel < 3:
        raise ValueError("vessel_preview_kernel must be >= 3")
    if config.stretch_percentiles is not None:
        low, high = config.stretch_percentiles
        if low < 0.0 or high > 100.0 or low >= high:
            raise ValueError(
                "stretch_percentiles must satisfy 0 <= low < high <= 100"
            )
    if not (0.0 < config.adaptive_roi_scale <= 1.0):
        raise ValueError("adaptive_roi_scale must satisfy 0 < scale <= 1")
    if not (0.0 < config.palm_core_width_ratio <= 1.0):
        raise ValueError("palm_core_width_ratio must satisfy 0 < ratio <= 1")


def validate_quality_filter_config(config: PalmQualityFilterConfig) -> None:
    if config.min_final_mean < 0.0:
        raise ValueError("min_final_mean must be >= 0")
    if config.min_final_p95 < 0.0:
        raise ValueError("min_final_p95 must be >= 0")
    if config.min_final_std < 0.0:
        raise ValueError("min_final_std must be >= 0")
    if not (0.0 <= config.max_dark_fraction_lt_110 <= 1.0):
        raise ValueError("max_dark_fraction_lt_110 must be between 0 and 1")
    if config.min_laplacian_var < 0.0:
        raise ValueError("min_laplacian_var must be >= 0")
    if config.min_gradient_p95 < 0.0:
        raise ValueError("min_gradient_p95 must be >= 0")
    if config.min_texture_to_gradient_ratio < 0.0:
        raise ValueError("min_texture_to_gradient_ratio must be >= 0")
    if not (0 <= config.dark_threshold <= 255):
        raise ValueError("dark_threshold must be between 0 and 255")
    if not (0 <= config.bright_threshold <= 255):
        raise ValueError("bright_threshold must be between 0 and 255")


def get_palm_mask(gray: np.ndarray) -> np.ndarray:
    """Segment the palm from a dark background using Otsu thresholding."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return keep_largest_component(mask)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the dominant hand/palm component after thresholding."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    if not contours:
        return cleaned

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(cleaned, [largest], -1, 255, thickness=cv2.FILLED)
    return cleaned


def contour_stats(mask: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, object]:
    """Return center and bounding box for the dominant hand contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image_shape
    if not contours:
        return {
            "found": False,
            "area": 0.0,
            "bbox": (0, 0, width, height),
            "center": (width // 2, height // 2),
        }

    largest = max(contours, key=cv2.contourArea)
    x, y, bbox_width, bbox_height = cv2.boundingRect(largest)
    moments = cv2.moments(largest)
    if moments["m00"] > 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    else:
        center_x = x + bbox_width // 2
        center_y = y + bbox_height // 2

    return {
        "found": True,
        "area": float(cv2.contourArea(largest)),
        "bbox": (int(x), int(y), int(x + bbox_width), int(y + bbox_height)),
        "center": (int(center_x), int(center_y)),
    }


def largest_true_run(flags: np.ndarray) -> Tuple[int, int] | None:
    """Return the [start, end) span of the longest contiguous True run."""
    best_start = None
    best_length = 0
    current_start = None

    for index, value in enumerate(flags):
        if value and current_start is None:
            current_start = index
        if (not value or index == len(flags) - 1) and current_start is not None:
            end = index + 1 if value and index == len(flags) - 1 else index
            run_length = end - current_start
            if run_length > best_length:
                best_start = current_start
                best_length = run_length
            current_start = None

    if best_start is None:
        return None
    return (int(best_start), int(best_start + best_length))


def palm_core_bbox(
    mask: np.ndarray,
    fallback_bbox: Tuple[int, int, int, int],
    width_ratio: float,
) -> Tuple[int, int, int, int]:
    """Estimate a palm-core box by keeping only rows with a broad hand mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return fallback_bbox

    row_widths = np.zeros(mask.shape[0], dtype=np.int32)
    for row_index in np.unique(ys):
        row_xs = xs[ys == row_index]
        row_widths[row_index] = int(row_xs.max() - row_xs.min() + 1)

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


def palm_contour_center(mask: np.ndarray) -> Tuple[int, int]:
    """Return the centroid of the largest detected palm contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        height, width = mask.shape
        return (width // 2, height // 2)

    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        height, width = mask.shape
        return (width // 2, height // 2)

    return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))


def intensity_weighted_centroid(
    gray: np.ndarray,
    mask: np.ndarray,
    rough_center: Tuple[int, int],
    window: int,
) -> Tuple[int, int]:
    """Refine the palm center by weighting brighter pixels inside the palm mask."""
    center_x, center_y = rough_center
    height, width = gray.shape

    x1 = max(0, center_x - window)
    x2 = min(width, center_x + window)
    y1 = max(0, center_y - window)
    y2 = min(height, center_y + window)

    patch = gray[y1:y2, x1:x2].astype(np.float64)
    mask_patch = mask[y1:y2, x1:x2].astype(np.float64) / 255.0

    weighted = patch * mask_patch
    total = weighted.sum() + 1e-9

    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    refined_x = int((xs * weighted).sum() / total) + x1
    refined_y = int((ys * weighted).sum() / total) + y1
    return (refined_x, refined_y)


def resolve_centroid_window(
    gray_shape: Tuple[int, int],
    roi_size: int,
    centroid_window: int,
) -> int:
    if centroid_window > 0:
        return int(centroid_window)

    height, width = gray_shape
    auto_window = max(roi_size // 2, 120)
    return min(auto_window, max(min(height, width) // 2, 1))


def clamp_square_box(
    center: Tuple[int, int],
    side: int,
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Build a square crop box and keep it inside image bounds."""
    height, width = image_shape
    side = int(min(max(1, side), width, height))
    half = side // 2
    center_x, center_y = center

    x1 = int(center_x - half)
    y1 = int(center_y - half)
    x2 = x1 + side
    y2 = y1 + side

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        x1 -= x2 - width
        x2 = width
    if y2 > height:
        y1 -= y2 - height
        y2 = height

    return (max(0, x1), max(0, y1), min(width, x2), min(height, y2))


def extract_roi(
    gray: np.ndarray,
    roi_size: int,
    centroid_window: int = 0,
    center_mode: str = "weighted",
    center_offset: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Extract a square ROI centered on the selected palm center."""
    palm_mask = get_palm_mask(gray)
    rough_center = palm_contour_center(palm_mask)
    resolved_window = resolve_centroid_window(gray.shape, roi_size, centroid_window)
    weighted_center = intensity_weighted_centroid(
        gray, palm_mask, rough_center, resolved_window
    )

    if center_mode == "weighted":
        selected_center = weighted_center
    elif center_mode == "contour":
        selected_center = rough_center
    else:
        raise ValueError("center_mode must be 'weighted' or 'contour'")

    center_before_offset = selected_center
    center_x = selected_center[0] + int(center_offset[0])
    center_y = selected_center[1] + int(center_offset[1])
    selected_center = (center_x, center_y)
    half = roi_size // 2
    height, width = gray.shape

    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    if x2 > width:
        x2 = width
        x1 = x2 - roi_size
    if y2 > height:
        y2 = height
        y1 = y2 - roi_size

    x1 = max(0, x1)
    y1 = max(0, y1)
    roi = gray[y1:y2, x1:x2]

    if roi.shape[0] < roi_size or roi.shape[1] < roi_size:
        padded = np.zeros((roi_size, roi_size), dtype=np.uint8)
        padded[: roi.shape[0], : roi.shape[1]] = roi
        roi = padded

    debug = {
        "rough_center": rough_center,
        "weighted_center": weighted_center,
        "center_before_offset": center_before_offset,
        "center_after_offset": selected_center,
        "refined_center": selected_center,
        "center_offset": (int(center_offset[0]), int(center_offset[1])),
        "center_mode": center_mode,
        "roi_box": (int(x1), int(y1), int(x2), int(y2)),
        "palm_mask": palm_mask,
        "centroid_window": int(resolved_window),
    }
    return roi, debug


def extract_adaptive_roi(
    gray: np.ndarray,
    roi_scale: float,
    width_ratio: float,
    centroid_window: int = 0,
    center_offset: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Extract a palm ROI sized from the detected palm core rather than a fixed box."""
    palm_mask = get_palm_mask(gray)
    stats = contour_stats(palm_mask, gray.shape)
    core_bbox = palm_core_bbox(palm_mask, stats["bbox"], width_ratio)

    x1, y1, x2, y2 = core_bbox
    bbox_width = max(1, x2 - x1)
    bbox_height = max(1, y2 - y1)
    side = int(round(min(bbox_width, bbox_height) * roi_scale))
    resolved_window = resolve_centroid_window(gray.shape, side, centroid_window)
    rough_center = (x1 + bbox_width // 2, y1 + bbox_height // 2)
    weighted_center = intensity_weighted_centroid(
        gray,
        palm_mask,
        rough_center,
        resolved_window,
    )
    center_before_offset = weighted_center
    refined_center = (
        weighted_center[0] + int(center_offset[0]),
        weighted_center[1] + int(center_offset[1]),
    )
    roi_box = clamp_square_box(refined_center, side, gray.shape)
    rx1, ry1, rx2, ry2 = roi_box
    roi = gray[ry1:ry2, rx1:rx2]

    debug = {
        "rough_center": rough_center,
        "weighted_center": weighted_center,
        "center_before_offset": center_before_offset,
        "center_after_offset": refined_center,
        "refined_center": refined_center,
        "center_offset": (int(center_offset[0]), int(center_offset[1])),
        "center_mode": "adaptive_weighted",
        "roi_box": roi_box,
        "palm_mask": palm_mask,
        "centroid_window": int(resolved_window),
        "adaptive_roi": True,
        "adaptive_roi_scale": float(roi_scale),
        "palm_core_width_ratio": float(width_ratio),
        "hand_bbox": stats["bbox"],
        "hand_center": stats["center"],
        "palm_bbox": core_bbox,
        "palm_area": float(stats["area"]),
        "roi_side": int(side),
    }
    return roi, debug


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float,
    tile_grid: Tuple[int, int],
) -> np.ndarray:
    """Enhance local contrast for palm-vein visibility."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def denoise_gray(gray: np.ndarray, h: float) -> np.ndarray:
    """Apply mild non-local means denoising when requested."""
    if h <= 0.0:
        return gray
    return cv2.fastNlMeansDenoising(
        gray,
        None,
        h=float(h),
        templateWindowSize=7,
        searchWindowSize=21,
    )


def normalize_and_resize(
    gray: np.ndarray,
    final_size: int,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> np.ndarray:
    """Normalize intensity to [0, 255] and resize to the final model-ready size."""
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.resize(
        normalized,
        (final_size, final_size),
        interpolation=interpolation,
    )


def resize_final(
    gray: np.ndarray,
    final_size: int,
    interpolation: int,
) -> np.ndarray:
    """Resize an already-normalized final image to model-ready dimensions."""
    return cv2.resize(
        gray,
        (final_size, final_size),
        interpolation=interpolation,
    )


def summarize_image_quality(gray: np.ndarray) -> Dict[str, object]:
    """Return compact grayscale quality metrics for tuning captures."""
    return {
        "mean": round(float(np.mean(gray)), 3),
        "std": round(float(np.std(gray)), 3),
        "min": int(np.min(gray)),
        "max": int(np.max(gray)),
        "p95": round(float(np.percentile(gray, 95)), 3),
        "p99": round(float(np.percentile(gray, 99)), 3),
        "dark_fraction": round(float(np.mean(gray <= 10)), 6),
        "saturated_fraction": round(float(np.mean(gray >= 250)), 6),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
    }


def summarize_final_quality(
    final: np.ndarray,
    config: PalmQualityFilterConfig,
) -> Dict[str, object]:
    """Return final-image metrics used by the usable-capture quality gate."""
    gradient_x = cv2.Sobel(final, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(final, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt((gradient_x * gradient_x) + (gradient_y * gradient_y))

    return {
        "mean": round(float(np.mean(final)), 3),
        "std": round(float(np.std(final)), 3),
        "p95": round(float(np.percentile(final, 95)), 3),
        "p99": round(float(np.percentile(final, 99)), 3),
        "dark_fraction_lt_110": round(
            float(np.mean(final < config.dark_threshold)), 6
        ),
        "bright_fraction_gt_220": round(
            float(np.mean(final > config.bright_threshold)), 6
        ),
        "laplacian_var": round(float(cv2.Laplacian(final, cv2.CV_64F).var()), 3),
        "gradient_p95": round(float(np.percentile(gradient, 95)), 3),
        "texture_to_gradient_ratio": round(
            float(cv2.Laplacian(final, cv2.CV_64F).var())
            / max(float(np.percentile(gradient, 95)), 1e-6),
            6,
        ),
    }


def quality_filter_thresholds(config: PalmQualityFilterConfig) -> Dict[str, object]:
    return {
        "min_final_mean": float(config.min_final_mean),
        "min_final_p95": float(config.min_final_p95),
        "min_final_std": float(config.min_final_std),
        "max_dark_fraction_lt_110": float(config.max_dark_fraction_lt_110),
        "min_laplacian_var": float(config.min_laplacian_var),
        "min_gradient_p95": float(config.min_gradient_p95),
        "min_texture_to_gradient_ratio": float(config.min_texture_to_gradient_ratio),
        "dark_threshold": int(config.dark_threshold),
        "bright_threshold": int(config.bright_threshold),
    }


def assess_palm_vein_quality(
    final: np.ndarray,
    config: PalmQualityFilterConfig | None = None,
) -> Dict[str, object]:
    """Assess whether a preprocessed final image has enough visible structure."""
    config = config or PalmQualityFilterConfig()
    validate_quality_filter_config(config)
    metrics = summarize_final_quality(final, config)
    reasons = []
    passed = 0
    checks = 0

    def check(condition: bool, reason: str) -> None:
        nonlocal passed, checks
        checks += 1
        if condition:
            passed += 1
        else:
            reasons.append(reason)

    check(metrics["mean"] >= config.min_final_mean, "final too dark")
    check(metrics["p95"] >= config.min_final_p95, "low bright-end contrast")
    check(metrics["std"] >= config.min_final_std, "low contrast")
    check(
        metrics["dark_fraction_lt_110"] <= config.max_dark_fraction_lt_110,
        "mostly dark final image",
    )
    check(metrics["laplacian_var"] >= config.min_laplacian_var, "low texture")
    check(metrics["gradient_p95"] >= config.min_gradient_p95, "weak palm/vein edges")
    check(
        metrics["texture_to_gradient_ratio"] >= config.min_texture_to_gradient_ratio,
        "edges dominate fine vessel texture",
    )

    score = passed / max(checks, 1)
    return {
        "usable": not reasons,
        "score": round(float(score), 6),
        "reasons": reasons,
        "metrics": metrics,
        "thresholds": quality_filter_thresholds(config),
    }


def percentile_stretch(
    gray: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Contrast-stretch with percentile clipping to avoid edge outliers dominating."""
    low = float(np.percentile(gray, low_percentile))
    high = float(np.percentile(gray, high_percentile))
    if high <= low:
        return np.zeros_like(gray, dtype=np.uint8)

    stretched = (gray.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def create_vessel_preview(
    gray: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """Create a diagnostic view where dark line-like structures become bright."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return percentile_stretch(blackhat)


def preprocessing_profile_settings(config: PalmPreprocessingConfig) -> Dict[str, object]:
    if config.profile == PROFILE_LEGACY:
        return {
            "center_mode": "weighted",
            "denoise_h": float(config.denoise_h),
            "resize_interpolation": cv2.INTER_LANCZOS4,
            "stretch_percentiles": config.stretch_percentiles,
            "adaptive_roi": bool(config.adaptive_roi),
        }
    if config.profile == PROFILE_CAPTURE_V2:
        return {
            "center_mode": "contour",
            "denoise_h": float(config.denoise_h),
            "resize_interpolation": cv2.INTER_AREA,
            "stretch_percentiles": config.stretch_percentiles,
            "adaptive_roi": bool(config.adaptive_roi),
        }
    if config.profile == PROFILE_DATASET_V3:
        return {
            "center_mode": "weighted",
            "denoise_h": float(config.denoise_h),
            "resize_interpolation": cv2.INTER_AREA,
            "stretch_percentiles": config.stretch_percentiles,
            "adaptive_roi": bool(config.adaptive_roi),
        }
    if config.profile == PROFILE_DATASET_V3_SOFT:
        return {
            "center_mode": "weighted",
            "denoise_h": float(config.denoise_h),
            "resize_interpolation": cv2.INTER_AREA,
            "stretch_percentiles": config.stretch_percentiles,
            "adaptive_roi": bool(config.adaptive_roi),
        }
    raise ValueError("Unsupported preprocessing profile")


def preprocess_palm_image(
    gray: np.ndarray,
    config: PalmPreprocessingConfig,
) -> Dict[str, object]:
    """Run ROI extraction, CLAHE, and resize on one grayscale palm image."""
    validate_preprocessing_config(config)
    settings = preprocessing_profile_settings(config)
    raw = gray.astype(np.uint8, copy=True)
    if bool(settings["adaptive_roi"]):
        roi, debug = extract_adaptive_roi(
            raw,
            roi_scale=float(config.adaptive_roi_scale),
            width_ratio=float(config.palm_core_width_ratio),
            centroid_window=config.centroid_window,
            center_offset=(int(config.center_offset_x), int(config.center_offset_y)),
        )
    else:
        roi, debug = extract_roi(
            raw,
            config.roi_size,
            config.centroid_window,
            center_mode=str(settings["center_mode"]),
            center_offset=(int(config.center_offset_x), int(config.center_offset_y)),
        )
    denoised = denoise_gray(roi, float(settings["denoise_h"]))
    clahe = apply_clahe(denoised, config.clahe_clip, config.clahe_tile)
    stretch_percentiles = settings["stretch_percentiles"]
    if stretch_percentiles is None:
        final_source = cv2.normalize(clahe, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
    else:
        final_source = percentile_stretch(
            clahe,
            float(stretch_percentiles[0]),
            float(stretch_percentiles[1]),
        )
    final = resize_final(
        final_source,
        config.final_size,
        interpolation=int(settings["resize_interpolation"]),
    )
    vessel_preview = create_vessel_preview(clahe, config.vessel_preview_kernel)
    quality_filter = assess_palm_vein_quality(final)

    debug.update(
        {
            "profile": config.profile,
            "roi_size": int(config.roi_size),
            "final_size": int(config.final_size),
            "clahe_clip": float(config.clahe_clip),
            "clahe_tile": (int(config.clahe_tile[0]), int(config.clahe_tile[1])),
            "denoise_h": float(settings["denoise_h"]),
            "vessel_preview_kernel": int(config.vessel_preview_kernel),
            "resize_interpolation": int(settings["resize_interpolation"]),
            "center_offset_x": int(config.center_offset_x),
            "center_offset_y": int(config.center_offset_y),
            "adaptive_roi": bool(config.adaptive_roi),
            "adaptive_roi_scale": float(config.adaptive_roi_scale),
            "palm_core_width_ratio": float(config.palm_core_width_ratio),
            "stretch_percentiles": (
                None
                if stretch_percentiles is None
                else [float(stretch_percentiles[0]), float(stretch_percentiles[1])]
            ),
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
        "mask": debug["palm_mask"],
        "denoised": denoised,
        "clahe": clahe,
        "final": final,
        "vessel_preview": vessel_preview,
        "debug": debug,
    }

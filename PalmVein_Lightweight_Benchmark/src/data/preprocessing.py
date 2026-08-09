"""Reference preprocessing functions; never run automatically by training.

The benchmark consumes the already-preprocessed dataset. These functions record
the ROI/CLAHE/normalization stages needed to reproduce a comparable dataset.
Exact acquisition-specific thresholds should be audited before reprocessing raw
images because reprocessing would create a different experimental condition.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        raise ValueError("No foreground component found for palm ROI")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == index, 255, 0).astype(np.uint8)


def extract_centered_square_roi(gray: np.ndarray, roi_size: int = 224) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError("Expected a grayscale image")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = largest_component(mask)
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        raise ValueError("Palm mask has zero area")
    cx, cy = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
    half = roi_size // 2
    padded = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_REFLECT_101)
    cx, cy = cx + half, cy + half
    return padded[cy - half : cy - half + roi_size, cx - half : cx - half + roi_size]


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)).apply(gray)


def normalize_uint8(gray: np.ndarray) -> np.ndarray:
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess_image(path: str | Path, output_size: int = 224) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    roi = extract_centered_square_roi(gray, roi_size=output_size)
    enhanced = apply_clahe(roi)
    normalized = normalize_uint8(enhanced)
    return cv2.resize(normalized, (output_size, output_size), interpolation=cv2.INTER_AREA)

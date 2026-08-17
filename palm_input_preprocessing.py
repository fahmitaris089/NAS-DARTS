"""Shared deterministic input preprocessing for palm-vein experiments.

This module is intentionally free of PyTorch dependencies so the exact same
implementation can be reused by training, ONNX calibration/evaluation, and the
Raspberry Pi benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
from PIL import Image


LEGACY_INPUT_PROFILE: Final[str] = "legacy"
ROBUST_PERCENTILE_V1: Final[str] = "robust_percentile_v1"
INPUT_PROFILES: Final[tuple[str, ...]] = (
    LEGACY_INPUT_PROFILE,
    ROBUST_PERCENTILE_V1,
)


@dataclass(frozen=True)
class RobustPercentileConfig:
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    minimum_dynamic_range: float = 16.0
    version: str = ROBUST_PERCENTILE_V1


ROBUST_PERCENTILE_CONFIG: Final[RobustPercentileConfig] = RobustPercentileConfig()


def input_profile_metadata(profile: str) -> dict[str, object]:
    validate_input_profile(profile)
    if profile == ROBUST_PERCENTILE_V1:
        return asdict(ROBUST_PERCENTILE_CONFIG)
    return {"version": LEGACY_INPUT_PROFILE}


def validate_input_profile(profile: str) -> None:
    if profile not in INPUT_PROFILES:
        raise ValueError(f"Unknown input profile {profile!r}; choose one of {INPUT_PROFILES}")


def robust_percentile_unit(
    gray: np.ndarray,
    config: RobustPercentileConfig = ROBUST_PERCENTILE_CONFIG,
) -> np.ndarray:
    """Map a grayscale image to float32 [0, 1] using the frozen v1 recipe."""
    array = np.asarray(gray)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image; received shape {array.shape}")
    if array.size == 0:
        raise ValueError("Cannot preprocess an empty image")
    values = array.astype(np.float32, copy=False)
    low = float(np.percentile(values, config.lower_percentile))
    high = float(np.percentile(values, config.upper_percentile))
    dynamic_range = max(high - low, config.minimum_dynamic_range)
    normalized = np.clip((values - low) / dynamic_range, 0.0, 1.0)
    return normalized.astype(np.float32, copy=False)


def grayscale_to_unit(gray: np.ndarray, profile: str = LEGACY_INPUT_PROFILE) -> np.ndarray:
    """Convert a grayscale array to the unit interval under ``profile``."""
    validate_input_profile(profile)
    if profile == ROBUST_PERCENTILE_V1:
        return robust_percentile_unit(gray)
    values = np.asarray(gray)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image; received shape {values.shape}")
    return (values.astype(np.float32) / 255.0).clip(0.0, 1.0)


def apply_input_profile_pil(image: Image.Image, profile: str) -> Image.Image:
    """Apply an input profile and return an 8-bit PIL grayscale image.

    Quantizing back to uint8 makes torchvision and NumPy/ONNX paths exactly
    reproducible to within one input intensity level.
    """
    validate_input_profile(profile)
    gray = image.convert("L")
    if profile == LEGACY_INPUT_PROFILE:
        return gray
    unit = robust_percentile_unit(np.asarray(gray, dtype=np.uint8))
    quantized = np.rint(unit * 255.0).astype(np.uint8)
    return Image.fromarray(quantized)


class ApplyInputProfile:
    """Torchvision-compatible PIL transform backed by the shared recipe."""

    def __init__(self, profile: str = LEGACY_INPUT_PROFILE) -> None:
        validate_input_profile(profile)
        self.profile = profile

    def __call__(self, image: Image.Image) -> Image.Image:
        return apply_input_profile_pil(image, self.profile)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(profile={self.profile!r})"


def preprocess_pil_to_imagenet_chw(
    image: Image.Image,
    input_size: int,
    profile: str = LEGACY_INPUT_PROFILE,
    *,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """PIL grayscale to normalized RGB CHW float32 for ONNX Runtime."""
    resized = image.convert("L").resize((input_size, input_size), Image.BILINEAR)
    profiled = apply_input_profile_pil(resized, profile)
    unit = np.asarray(profiled, dtype=np.float32) / 255.0
    rgb = np.stack([unit, unit, unit], axis=0)
    mean_array = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_array = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return ((rgb - mean_array) / std_array).astype(np.float32)


def preprocess_path_to_imagenet_bchw(
    path: str,
    input_size: int,
    profile: str = LEGACY_INPUT_PROFILE,
) -> np.ndarray:
    with Image.open(path) as image:
        chw = preprocess_pil_to_imagenet_chw(image, input_size, profile)
    return np.expand_dims(chw, axis=0)

"""Auto-capture palm images on Raspberry Pi when a hand is detected.

This script is intended to run directly on Raspberry Pi OS with Picamera2.
It watches the camera feed, learns an empty-scanner background, detects a
large foreground object that looks palm-like, captures a short burst once the
object is stable and centered for several consecutive frames, and saves the
best frame from that burst. After a capture event, the scanner waits until the
object is removed before it arms itself again.

Recommended install on Raspberry Pi:
    sudo apt update
    sudo apt install -y python3-picamera2 python3-opencv

Example:
    python3 capture_on_hand_detect.py --out-dir captures --preview

Important:
    Stop other camera users first, such as libcamera-vid or VLC pipelines.
    Picamera2 needs direct access to the camera device.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from palm_preprocessing import (
    PROFILE_CAPTURE_V2,
    PROFILE_DATASET_V3,
    PROFILE_DATASET_V3_SOFT,
    PROFILE_LEGACY,
    SUPPORTED_PROFILES,
    PalmPreprocessingConfig,
    PalmQualityFilterConfig,
    assess_palm_vein_quality,
    preprocess_palm_image,
)

try:
    from picamera2 import Picamera2
except ImportError as exc:
    raise SystemExit(
        "Picamera2 is not installed. Install it on Raspberry Pi with: "
        "sudo apt install -y python3-picamera2"
    ) from exc


BURST_SCORE_VERSION = "quality_v1"
BURST_SCORE_WEIGHTS = {
    "sharpness": 0.45,
    "contrast": 0.35,
    "saturation": 0.15,
    "centering": 0.05,
}


def parse_denoise_mode(text: str) -> str:
    normalized = text.strip().lower().replace("-", "_")
    aliases = {
        "off": "off",
        "none": "off",
        "false": "off",
        "fast": "fast",
        "minimal": "minimal",
        "high_quality": "high_quality",
        "highquality": "high_quality",
        "hq": "high_quality",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            "Invalid denoise mode. Use one of: off, fast, minimal, high_quality."
        )
    return aliases[normalized]


def parse_awbgains(text: str) -> Tuple[float, float]:
    try:
        red_text, blue_text = text.split(",", maxsplit=1)
        red_gain = float(red_text)
        blue_gain = float(blue_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid awb gains '{text}'. Use RED,BLUE, for example 1.0,1.0."
        ) from exc

    if red_gain <= 0.0 or blue_gain <= 0.0:
        raise argparse.ArgumentTypeError("AWB gains must be positive.")
    return red_gain, blue_gain


def parse_size(text: str) -> Tuple[int, int]:
    try:
        width_text, height_text = text.lower().split("x", maxsplit=1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid size '{text}'. Use WIDTHxHEIGHT, for example 1280x720."
        ) from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive.")
    return width, height


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto capture when hand is detected")
    parser.add_argument("--out-dir", default="captures", help="Directory for saved images")
    parser.add_argument(
        "--size",
        type=parse_size,
        default=(1280, 720),
        help="Preview/capture size, e.g. 1280x720",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=2.0,
        help="Camera warmup time before background learning",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target preview FPS. Lower FPS allows longer exposure in dark NIR scenes.",
    )
    parser.add_argument(
        "--frame-duration-us",
        type=int,
        default=0,
        help="Fixed frame duration in microseconds. 0 derives it from FPS.",
    )
    parser.add_argument(
        "--background-frames",
        type=int,
        default=30,
        help="Number of empty-scanner frames used to learn background",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=7,
        help="Gaussian blur kernel size, odd number",
    )
    parser.add_argument(
        "--diff-threshold",
        type=int,
        default=25,
        help="Threshold on absolute difference from background",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=7,
        help="Morphology kernel size, odd number",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.04,
        help="Minimum contour area ratio relative to full frame",
    )
    parser.add_argument(
        "--capture-zone-ratio",
        type=float,
        default=0.60,
        help="Center box ratio that hand center must enter before capture",
    )
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=0.55,
        help="Minimum bbox width/height ratio for a palm-like object",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=1.85,
        help="Maximum bbox width/height ratio for a palm-like object",
    )
    parser.add_argument(
        "--min-extent",
        type=float,
        default=0.38,
        help="Minimum contour extent (area / bbox area) for a palm-like object",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=8,
        help="Consecutive valid frames required before capture",
    )
    parser.add_argument(
        "--burst-frames",
        type=int,
        default=3,
        help="Frames to capture per trigger; the best frame is saved",
    )
    parser.add_argument(
        "--rearm-empty-frames",
        type=int,
        default=8,
        help="Empty frames required before the scanner can capture again",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=2.0,
        help="Minimum delay between separate capture events",
    )
    parser.add_argument(
        "--background-update-rate",
        type=float,
        default=0.02,
        help="Background update rate when no hand is present",
    )
    parser.add_argument(
        "--exposure-us",
        "--shutter",
        type=int,
        dest="exposure_us",
        default=0,
        help="Fixed exposure/shutter in microseconds, 0 keeps auto exposure",
    )
    parser.add_argument(
        "--gain",
        "--analoggain",
        type=float,
        dest="gain",
        default=0.0,
        help="Fixed analogue gain, 0 keeps auto gain",
    )
    parser.add_argument(
        "--awbgains",
        type=parse_awbgains,
        default=None,
        metavar="RED,BLUE",
        help=(
            "Manual AWB gains as RED,BLUE, for example 1.0,1.0. "
            "Setting this disables auto white balance."
        ),
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=None,
        help="Manual brightness control, for example -0.1.",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=None,
        help="Manual contrast control, for example 1.1.",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=None,
        help="Manual saturation control, for example 0.",
    )
    parser.add_argument(
        "--denoise",
        type=parse_denoise_mode,
        default=None,
        help="Camera noise reduction mode: off, fast, minimal, or high_quality.",
    )
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="Use less strict palm detection while still requiring centered placement",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show OpenCV preview window with overlays",
    )
    parser.add_argument(
        "--preview-autostretch",
        action="store_true",
        help="Stretch preview contrast to make dark NIR scenes easier to inspect.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run palm preprocessing after each capture and save ROI plus final outputs",
    )
    parser.add_argument(
        "--preprocess-profile",
        choices=SUPPORTED_PROFILES,
        default=PROFILE_DATASET_V3,
        help=(
            "Preprocessing profile. legacy keeps the original model-compatible "
            "pipeline; capture_v2 uses contour-centered ROI, mild denoise, "
            "conservative CLAHE, and area downsampling; dataset_v3 is the "
            "default for recent 1920x1080 captures; dataset_v3_soft keeps the "
            "same adaptive ROI but softens local enhancement for sharper raw inputs."
        ),
    )
    parser.add_argument(
        "--preprocess-roi-size",
        type=int,
        default=384,
        help="Square ROI size for preprocessing before final resize",
    )
    parser.add_argument(
        "--preprocess-final-size",
        type=int,
        default=224,
        help="Final preprocessed image size",
    )
    parser.add_argument(
        "--preprocess-clahe-clip",
        type=float,
        default=2.0,
        help="CLAHE clip limit used during preprocessing",
    )
    parser.add_argument(
        "--preprocess-clahe-tile",
        type=parse_size,
        default=(8, 8),
        help="CLAHE tile grid size as WIDTHxHEIGHT, for example 8x8",
    )
    parser.add_argument(
        "--preprocess-centroid-window",
        type=int,
        default=0,
        help="Centroid refinement window. 0 chooses an automatic window.",
    )
    parser.add_argument(
        "--preprocess-denoise-h",
        type=float,
        default=None,
        help=(
            "Non-local means denoise strength before CLAHE. If omitted, legacy "
            "uses 0.0, capture_v2 uses 5.0, and dataset_v3 uses 3.0."
        ),
    )
    parser.add_argument(
        "--preprocess-center-offset-x",
        type=int,
        default=0,
        help="Horizontal ROI center offset in pixels after palm center detection.",
    )
    parser.add_argument(
        "--preprocess-center-offset-y",
        type=int,
        default=0,
        help="Vertical ROI center offset in pixels after palm center detection.",
    )
    parser.add_argument(
        "--preprocess-stretch-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help=(
            "Percentiles used to contrast-stretch CLAHE output before final "
            "resize. Omit to use min-max normalization."
        ),
    )
    parser.add_argument(
        "--preprocess-adaptive-roi",
        action="store_true",
        help=(
            "Use adaptive palm-core ROI sizing integrated from preprocess_v2 "
            "instead of a fixed square ROI."
        ),
    )
    parser.add_argument(
        "--preprocess-adaptive-roi-scale",
        type=float,
        default=0.95,
        help="Adaptive ROI side as a fraction of the detected palm-core span.",
    )
    parser.add_argument(
        "--preprocess-palm-core-width-ratio",
        type=float,
        default=0.45,
        help="Rows at least this wide fraction of the max hand width define the palm core.",
    )
    parser.add_argument(
        "--preprocess-vessel-kernel",
        type=int,
        default=17,
        help="Odd morphology kernel size for diagnostic vessel preview output.",
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help=(
            "Reject captures after preprocessing when the final image is too dark, "
            "flat, or lacks enough palm/vein texture. Requires --preprocess."
        ),
    )
    parser.add_argument(
        "--quality-min-laplacian-var",
        type=float,
        default=24.0,
        help="Minimum Laplacian variance required by the quality filter.",
    )
    parser.add_argument(
        "--save-rejected",
        action="store_true",
        help=(
            "When --quality-filter rejects a capture, save it under rejected/ for "
            "debugging instead of discarding it completely."
        ),
    )
    return parser.parse_args(argv)


def collect_explicit_options(argv: Sequence[str], tracked_options: Set[str]) -> Set[str]:
    explicit_options: Set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("--"):
            continue
        option = token.split("=", maxsplit=1)[0]
        if option in tracked_options:
            explicit_options.add(option)
    return explicit_options


def apply_relaxed_preset(args: argparse.Namespace, explicit_options: Set[str]) -> None:
    if not args.relaxed:
        return

    relaxed_values = {
        "--capture-zone-ratio": 0.80,
        "--stable-frames": 3,
        "--min-extent": 0.30,
        "--min-aspect-ratio": 0.45,
        "--max-aspect-ratio": 2.15,
    }

    for option, value in relaxed_values.items():
        if option in explicit_options:
            continue
        attr_name = option.lstrip("-").replace("-", "_")
        setattr(args, attr_name, value)


def has_gui_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def ensure_odd(value: int, name: str) -> int:
    if value < 3:
        return 3
    if value % 2 == 0:
        value += 1
    return value


def configure_camera(args: argparse.Namespace) -> Picamera2:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": args.size, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    controls: Dict[str, object] = {}
    if args.frame_duration_us > 0:
        frame_duration_us = args.frame_duration_us
    else:
        frame_duration_us = int(1_000_000 / max(args.fps, 1.0))

    controls["FrameDurationLimits"] = (frame_duration_us, frame_duration_us)

    if args.awbgains is not None:
        controls["AwbEnable"] = False
        controls["ColourGains"] = (
            float(args.awbgains[0]),
            float(args.awbgains[1]),
        )

    if args.exposure_us > 0 or args.gain > 0:
        controls["AeEnable"] = False
        if args.exposure_us > 0:
            controls["ExposureTime"] = args.exposure_us
            if args.exposure_us > frame_duration_us:
                controls["FrameDurationLimits"] = (args.exposure_us, args.exposure_us)
        if args.gain > 0:
            controls["AnalogueGain"] = args.gain

    if args.brightness is not None:
        controls["Brightness"] = float(args.brightness)
    if args.contrast is not None:
        controls["Contrast"] = float(args.contrast)
    if args.saturation is not None:
        controls["Saturation"] = float(args.saturation)
    if args.denoise is not None:
        denoise_modes = {
            "off": 0,
            "fast": 1,
            "high_quality": 2,
            "minimal": 3,
        }
        controls["NoiseReductionMode"] = denoise_modes[str(args.denoise)]
    if controls:
        picam2.set_controls(controls)

    time.sleep(args.warmup_seconds)
    return picam2


def resolve_output_dirs(
    output_root: Path,
    preprocess_enabled: bool,
) -> Tuple[Path, Optional[Path]]:
    if preprocess_enabled:
        return output_root / "raw", output_root / "processed"
    return output_root, None


def resolve_rejected_output_dirs(
    output_root: Path,
    preprocess_enabled: bool,
) -> Tuple[Path, Optional[Path]]:
    rejected_root = output_root / "rejected"
    if preprocess_enabled:
        return rejected_root / "raw", rejected_root / "processed"
    return rejected_root, None


def relative_output_path(path: Path, output_root: Path) -> str:
    try:
        return str(path.relative_to(output_root))
    except ValueError:
        return path.name


def write_gray_image(image_path: Path, gray: np.ndarray) -> None:
    if not cv2.imwrite(str(image_path), gray):
        raise OSError(f"Failed to write image: {image_path}")


def build_preprocessing_config(
    args: argparse.Namespace,
    explicit_options: Set[str],
) -> PalmPreprocessingConfig:
    roi_size = int(args.preprocess_roi_size)
    clahe_clip = float(args.preprocess_clahe_clip)
    clahe_tile = (
        int(args.preprocess_clahe_tile[0]),
        int(args.preprocess_clahe_tile[1]),
    )
    denoise_h = args.preprocess_denoise_h
    center_offset_x = int(args.preprocess_center_offset_x)
    center_offset_y = int(args.preprocess_center_offset_y)
    stretch_percentiles = args.preprocess_stretch_percentiles
    adaptive_roi = bool(args.preprocess_adaptive_roi)
    adaptive_roi_scale = float(args.preprocess_adaptive_roi_scale)
    palm_core_width_ratio = float(args.preprocess_palm_core_width_ratio)

    if args.preprocess_profile == PROFILE_CAPTURE_V2:
        if "--preprocess-clahe-clip" not in explicit_options:
            clahe_clip = 1.2
        if "--preprocess-clahe-tile" not in explicit_options:
            clahe_tile = (12, 12)
        if denoise_h is None:
            denoise_h = 5.0
    elif args.preprocess_profile == PROFILE_DATASET_V3:
        if "--preprocess-adaptive-roi" not in explicit_options:
            adaptive_roi = True
        if "--preprocess-roi-size" not in explicit_options:
            roi_size = 760
        if "--preprocess-clahe-clip" not in explicit_options:
            clahe_clip = 2.4
        if "--preprocess-clahe-tile" not in explicit_options:
            clahe_tile = (8, 8)
        if "--preprocess-center-offset-x" not in explicit_options:
            center_offset_x = 0
        if "--preprocess-center-offset-y" not in explicit_options:
            center_offset_y = 0
        if "--preprocess-stretch-percentiles" not in explicit_options:
            stretch_percentiles = None
        if "--preprocess-adaptive-roi-scale" not in explicit_options:
            adaptive_roi_scale = 0.95
        if "--preprocess-palm-core-width-ratio" not in explicit_options:
            palm_core_width_ratio = 0.45
        if denoise_h is None:
            denoise_h = 0.0
    elif args.preprocess_profile == PROFILE_DATASET_V3_SOFT:
        if "--preprocess-adaptive-roi" not in explicit_options:
            adaptive_roi = True
        if "--preprocess-roi-size" not in explicit_options:
            roi_size = 760
        if "--preprocess-clahe-clip" not in explicit_options:
            clahe_clip = 1.6
        if "--preprocess-clahe-tile" not in explicit_options:
            clahe_tile = (12, 12)
        if "--preprocess-center-offset-x" not in explicit_options:
            center_offset_x = 0
        if "--preprocess-center-offset-y" not in explicit_options:
            center_offset_y = 0
        if "--preprocess-stretch-percentiles" not in explicit_options:
            stretch_percentiles = None
        if "--preprocess-adaptive-roi-scale" not in explicit_options:
            adaptive_roi_scale = 0.95
        if "--preprocess-palm-core-width-ratio" not in explicit_options:
            palm_core_width_ratio = 0.45
        if denoise_h is None:
            denoise_h = 2.0
    elif denoise_h is None:
        denoise_h = 0.0

    if stretch_percentiles is not None:
        stretch_percentiles = (
            float(stretch_percentiles[0]),
            float(stretch_percentiles[1]),
        )

    return PalmPreprocessingConfig(
        roi_size=roi_size,
        final_size=int(args.preprocess_final_size),
        clahe_clip=clahe_clip,
        clahe_tile=clahe_tile,
        centroid_window=int(args.preprocess_centroid_window),
        profile=str(args.preprocess_profile),
        denoise_h=float(denoise_h),
        vessel_preview_kernel=int(args.preprocess_vessel_kernel),
        center_offset_x=center_offset_x,
        center_offset_y=center_offset_y,
        stretch_percentiles=stretch_percentiles,
        adaptive_roi=adaptive_roi,
        adaptive_roi_scale=adaptive_roi_scale,
        palm_core_width_ratio=palm_core_width_ratio,
    )


def preprocessing_parameter_metadata(
    config: PalmPreprocessingConfig,
) -> Dict[str, object]:
    return {
        "profile": str(config.profile),
        "roi_size": int(config.roi_size),
        "final_size": int(config.final_size),
        "clahe_clip": float(config.clahe_clip),
        "clahe_tile": [int(config.clahe_tile[0]), int(config.clahe_tile[1])],
        "centroid_window": int(config.centroid_window),
        "denoise_h": float(config.denoise_h),
        "vessel_preview_kernel": int(config.vessel_preview_kernel),
        "center_offset_x": int(config.center_offset_x),
        "center_offset_y": int(config.center_offset_y),
        "adaptive_roi": bool(config.adaptive_roi),
        "adaptive_roi_scale": float(config.adaptive_roi_scale),
        "palm_core_width_ratio": float(config.palm_core_width_ratio),
        "stretch_percentiles": (
            None
            if config.stretch_percentiles is None
            else [
                float(config.stretch_percentiles[0]),
                float(config.stretch_percentiles[1]),
            ]
        ),
    }


def build_quality_filter_config(args: argparse.Namespace) -> PalmQualityFilterConfig:
    return PalmQualityFilterConfig(
        min_laplacian_var=float(args.quality_min_laplacian_var),
    )


def save_preprocessed_outputs(
    processed_dir: Path,
    capture_stem: str,
    preprocessing_result: Dict[str, object],
) -> Dict[str, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_root = processed_dir.parent
    final_dir = output_root / "final"
    visualization_dir = output_root / "visualizations"

    final_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    roi_path = processed_dir / f"{capture_stem}_roi.png"
    final_path = final_dir / f"{capture_stem}_final.png"
    mask_path = processed_dir / f"{capture_stem}_mask.png"
    clahe_path = processed_dir / f"{capture_stem}_clahe.png"
    vessel_preview_path = visualization_dir / f"{capture_stem}_vessel_preview.png"

    write_gray_image(roi_path, preprocessing_result["roi"])
    write_gray_image(final_path, preprocessing_result["final"])
    write_gray_image(mask_path, preprocessing_result["mask"])
    write_gray_image(clahe_path, preprocessing_result["clahe"])
    write_gray_image(vessel_preview_path, preprocessing_result["vessel_preview"])

    return {
        "roi_path": roi_path,
        "final_path": final_path,
        "mask_path": mask_path,
        "clahe_path": clahe_path,
        "vessel_preview_path": vessel_preview_path,
    }

def build_preprocessing_metadata(
    output_root: Path,
    processed_paths: Dict[str, Path],
    preprocessing_result: Dict[str, object],
    config: PalmPreprocessingConfig,
) -> Dict[str, object]:
    debug = preprocessing_result["debug"]
    metadata = {
        "enabled": True,
        "ran": True,
        "roi_image": processed_paths["roi_path"].name,
        "roi_relative_path": relative_output_path(processed_paths["roi_path"], output_root),
        "final_image": processed_paths["final_path"].name,
        "final_relative_path": relative_output_path(
            processed_paths["final_path"],
            output_root,
        ),
        "mask_image": processed_paths["mask_path"].name,
        "mask_relative_path": relative_output_path(
            processed_paths["mask_path"],
            output_root,
        ),
        "clahe_image": processed_paths["clahe_path"].name,
        "clahe_relative_path": relative_output_path(
            processed_paths["clahe_path"],
            output_root,
        ),
        "vessel_preview_image": processed_paths["vessel_preview_path"].name,
        "vessel_preview_relative_path": relative_output_path(
            processed_paths["vessel_preview_path"],
            output_root,
        ),
        "rough_center": [
            int(debug["rough_center"][0]),
            int(debug["rough_center"][1]),
        ],
        "weighted_center": [
            int(debug["weighted_center"][0]),
            int(debug["weighted_center"][1]),
        ],
        "center_before_offset": [
            int(debug["center_before_offset"][0]),
            int(debug["center_before_offset"][1]),
        ],
        "center_after_offset": [
            int(debug["center_after_offset"][0]),
            int(debug["center_after_offset"][1]),
        ],
        "refined_center": [
            int(debug["refined_center"][0]),
            int(debug["refined_center"][1]),
        ],
        "center_offset_x": int(debug["center_offset"][0]),
        "center_offset_y": int(debug["center_offset"][1]),
        "center_mode": str(debug["center_mode"]),
        "roi_box": [int(value) for value in debug["roi_box"]],
        "centroid_window": int(debug["centroid_window"]),
        "quality": debug["quality"],
    }
    metadata.update(preprocessing_parameter_metadata(config))
    return metadata


def build_preprocessing_failure_metadata(
    config: PalmPreprocessingConfig,
    error: str,
) -> Dict[str, object]:
    metadata = {
        "enabled": True,
        "ran": False,
        "error": error,
    }
    metadata.update(preprocessing_parameter_metadata(config))
    return metadata


def build_quality_filter_metadata(
    preprocessing_result: Dict[str, object],
    enabled: bool,
    config: PalmQualityFilterConfig,
) -> Dict[str, object]:
    result = assess_palm_vein_quality(preprocessing_result["final"], config)
    return {
        "enabled": bool(enabled),
        "usable": bool(result["usable"]),
        "score": float(result["score"]),
        "reasons": [str(reason) for reason in result["reasons"]],
        "metrics": result["metrics"],
        "thresholds": result["thresholds"],
    }


def quality_filter_rejection_message(metadata: Dict[str, object]) -> str:
    reasons = metadata.get("reasons", [])
    if reasons:
        reason_text = ", ".join(str(reason) for reason in reasons)
    else:
        reason_text = "quality score below threshold"
    return f"Rejected capture: vein pattern not usable ({reason_text})."


def capture_gray_frame(picam2: Picamera2) -> Tuple[np.ndarray, np.ndarray]:
    frame_rgb = picam2.capture_array("main")
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return frame_rgb, gray


def build_background(
    picam2: Picamera2,
    background_frames: int,
    blur_kernel: int,
) -> np.ndarray:
    print("Learning background. Keep scanner empty for a moment...")
    accumulator = None

    for index in range(background_frames):
        _, gray = capture_gray_frame(picam2)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        blurred_f32 = blurred.astype(np.float32)

        if accumulator is None:
            accumulator = blurred_f32
        else:
            accumulator += blurred_f32

        if (index + 1) % 10 == 0 or index == background_frames - 1:
            print(f"  background frame {index + 1}/{background_frames}")

    background = accumulator / float(background_frames)
    return background.astype(np.uint8)


def center_capture_zone(shape: Tuple[int, int], ratio: float) -> Tuple[int, int, int, int]:
    height, width = shape
    zone_width = int(width * ratio)
    zone_height = int(height * ratio)
    x1 = (width - zone_width) // 2
    y1 = (height - zone_height) // 2
    x2 = x1 + zone_width
    y2 = y1 + zone_height
    return x1, y1, x2, y2


def detect_hand(
    gray: np.ndarray,
    background: np.ndarray,
    blur_kernel: int,
    diff_threshold: int,
    morph_kernel: int,
    min_area: float,
    capture_zone_ratio: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    min_extent: float,
) -> Tuple[Optional[Dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    diff = cv2.absdiff(blurred, background)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, blurred, diff, mask

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < min_area:
        return None, blurred, diff, mask

    x, y, w, h = cv2.boundingRect(largest)
    center_x = x + (w // 2)
    center_y = y + (h // 2)
    bbox_area = float(w * h)
    aspect_ratio = float(w) / max(float(h), 1.0)
    extent = area / max(bbox_area, 1.0)

    zone_x1, zone_y1, zone_x2, zone_y2 = center_capture_zone(gray.shape, capture_zone_ratio)
    centered = zone_x1 <= center_x <= zone_x2 and zone_y1 <= center_y <= zone_y2
    hand_like = min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and extent >= min_extent

    info = {
        "area": area,
        "bbox": (int(x), int(y), int(w), int(h)),
        "center": (int(center_x), int(center_y)),
        "centered": centered,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "hand_like": hand_like,
        "zone": (int(zone_x1), int(zone_y1), int(zone_x2), int(zone_y2)),
    }
    return info, blurred, diff, mask


def summarize_brightness(gray: np.ndarray) -> Dict[str, object]:
    mean_gray = float(np.mean(gray))
    std_gray = float(np.std(gray))
    min_gray = int(np.min(gray))
    max_gray = int(np.max(gray))
    p95_gray = float(np.percentile(gray, 95))
    p99_gray = float(np.percentile(gray, 99))
    saturated_fraction = float(np.mean(gray >= 250))
    return {
        "mean_gray": round(mean_gray, 3),
        "std_gray": round(std_gray, 3),
        "min_gray": min_gray,
        "max_gray": max_gray,
        "p95_gray": round(p95_gray, 3),
        "p99_gray": round(p99_gray, 3),
        "saturated_fraction": round(saturated_fraction, 6),
    }


def compute_frame_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_center_distance(
    detection: Optional[Dict[str, object]],
) -> Optional[float]:
    if detection is None:
        return None

    center_x, center_y = detection["center"]
    zone_x1, zone_y1, zone_x2, zone_y2 = detection["zone"]
    zone_center_x = (zone_x1 + zone_x2) / 2.0
    zone_center_y = (zone_y1 + zone_y2) / 2.0
    zone_width = max(float(zone_x2 - zone_x1), 1.0)
    zone_height = max(float(zone_y2 - zone_y1), 1.0)
    zone_diagonal = max(float(np.hypot(zone_width, zone_height)), 1.0)
    distance = float(np.hypot(center_x - zone_center_x, center_y - zone_center_y))
    return distance / zone_diagonal


def build_burst_candidate(
    frame_index: int,
    gray: np.ndarray,
    detection: Optional[Dict[str, object]],
) -> Dict[str, object]:
    brightness = summarize_brightness(gray)
    sharpness = compute_frame_sharpness(gray)
    center_distance = compute_center_distance(detection)
    hand_like = bool(detection is not None and detection["hand_like"])
    centered = bool(detection is not None and detection["centered"])
    return {
        "frame_index": frame_index,
        "gray": gray,
        "detection": detection,
        "brightness": brightness,
        "sharpness": sharpness,
        "center_distance": center_distance,
        "hand_like": hand_like,
        "centered": centered,
    }


def normalize_burst_metric(
    values: Sequence[Optional[float]],
    *,
    prefer_higher: bool,
    missing_score: float = 0.0,
) -> Tuple[float, ...]:
    present_values = [float(value) for value in values if value is not None]
    if not present_values:
        return tuple(float(missing_score) for _ in values)

    low = min(present_values)
    high = max(present_values)
    if high - low < 1e-9:
        return tuple(
            0.5 if value is not None else float(missing_score) for value in values
        )

    normalized_scores = []
    for value in values:
        if value is None:
            normalized_scores.append(float(missing_score))
            continue

        score = (float(value) - low) / (high - low)
        if not prefer_higher:
            score = 1.0 - score
        normalized_scores.append(score)
    return tuple(normalized_scores)


def attach_burst_quality_scores(candidates: Sequence[Dict[str, object]]) -> None:
    sharpness_scores = normalize_burst_metric(
        [float(candidate["sharpness"]) for candidate in candidates],
        prefer_higher=True,
    )
    contrast_scores = normalize_burst_metric(
        [float(candidate["brightness"]["std_gray"]) for candidate in candidates],
        prefer_higher=True,
    )
    saturation_scores = normalize_burst_metric(
        [float(candidate["brightness"]["saturated_fraction"]) for candidate in candidates],
        prefer_higher=False,
    )
    centering_scores = normalize_burst_metric(
        [
            float(candidate["center_distance"])
            if candidate["center_distance"] is not None
            else None
            for candidate in candidates
        ],
        prefer_higher=False,
        missing_score=0.0,
    )

    for candidate, sharpness_score, contrast_score, saturation_score, centering_score in zip(
        candidates,
        sharpness_scores,
        contrast_scores,
        saturation_scores,
        centering_scores,
    ):
        quality_score = (
            BURST_SCORE_WEIGHTS["sharpness"] * sharpness_score
            + BURST_SCORE_WEIGHTS["contrast"] * contrast_score
            + BURST_SCORE_WEIGHTS["saturation"] * saturation_score
            + BURST_SCORE_WEIGHTS["centering"] * centering_score
        )
        candidate["burst_score_version"] = BURST_SCORE_VERSION
        candidate["burst_score"] = quality_score
        candidate["burst_score_components"] = {
            "sharpness_score": round(sharpness_score, 6),
            "contrast_score": round(contrast_score, 6),
            "saturation_score": round(saturation_score, 6),
            "centering_score": round(centering_score, 6),
        }


def burst_candidate_sort_key(candidate: Dict[str, object]) -> Tuple[float, ...]:
    brightness = candidate["brightness"]
    center_distance = candidate["center_distance"]
    return (
        float(int(candidate["hand_like"] and candidate["centered"])),
        float(int(candidate["hand_like"])),
        float(int(candidate["centered"])),
        float(candidate.get("burst_score", 0.0)),
        float(candidate["sharpness"]),
        float(brightness["std_gray"]),
        -float(brightness["saturated_fraction"]),
        -float(center_distance if center_distance is not None else 1.0),
        -float(candidate["frame_index"]),
    )


def summarize_burst_candidate(candidate: Dict[str, object]) -> Dict[str, object]:
    summary = {
        "frame_index": int(candidate["frame_index"]),
        "hand_like": bool(candidate["hand_like"]),
        "centered": bool(candidate["centered"]),
        "sharpness": round(float(candidate["sharpness"]), 3),
        "burst_score": round(float(candidate.get("burst_score", 0.0)), 6),
        "brightness": candidate["brightness"],
    }
    if "burst_score_version" in candidate:
        summary["burst_score_version"] = str(candidate["burst_score_version"])
    if "burst_score_components" in candidate:
        summary["burst_score_components"] = candidate["burst_score_components"]
    if candidate["center_distance"] is not None:
        summary["center_distance"] = round(float(candidate["center_distance"]), 6)
    else:
        summary["center_distance"] = None
    return summary


def capture_burst_best_frame(
    picam2: Picamera2,
    initial_gray: np.ndarray,
    initial_detection: Dict[str, object],
    background: np.ndarray,
    args: argparse.Namespace,
    min_area: float,
) -> Tuple[np.ndarray, Dict[str, object], Dict[str, object]]:
    candidates = [build_burst_candidate(0, initial_gray, initial_detection)]

    for frame_index in range(1, args.burst_frames):
        _, burst_gray = capture_gray_frame(picam2)
        burst_detection, _, _, _ = detect_hand(
            gray=burst_gray,
            background=background,
            blur_kernel=args.blur_kernel,
            diff_threshold=args.diff_threshold,
            morph_kernel=args.morph_kernel,
            min_area=min_area,
            capture_zone_ratio=args.capture_zone_ratio,
            min_aspect_ratio=args.min_aspect_ratio,
            max_aspect_ratio=args.max_aspect_ratio,
            min_extent=args.min_extent,
        )
        candidates.append(build_burst_candidate(frame_index, burst_gray, burst_detection))

    attach_burst_quality_scores(candidates)
    best_candidate = max(candidates, key=burst_candidate_sort_key)
    best_detection = best_candidate["detection"]
    if best_detection is None:
        best_detection = initial_detection

    burst_summary = {
        "burst_frames": int(args.burst_frames),
        "selected_frame_index": int(best_candidate["frame_index"]),
        "selected_burst_score": round(float(best_candidate.get("burst_score", 0.0)), 6),
        "burst_score_version": BURST_SCORE_VERSION,
        "score_weights": BURST_SCORE_WEIGHTS,
        "selection_metric": (
            "gate on centered hand-like frames, then prefer higher burst quality "
            "score driven mainly by sharpness and gray-level contrast, penalize "
            "saturation, and use center distance as a weak tie-breaker"
        ),
        "frames": [summarize_burst_candidate(candidate) for candidate in candidates],
    }
    return best_candidate["gray"], best_detection, burst_summary


def save_capture(
    out_dir: Path,
    output_root: Path,
    gray: np.ndarray,
    detection: Dict[str, object],
    args: argparse.Namespace,
    frame_duration_us: int,
    burst_summary: Optional[Dict[str, object]] = None,
    preprocessing_metadata: Optional[Dict[str, object]] = None,
    quality_filter_metadata: Optional[Dict[str, object]] = None,
) -> Tuple[Path, Path, Dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = out_dir / f"palm_{timestamp}.png"
    meta_path = out_dir / f"palm_{timestamp}.json"

    write_gray_image(image_path, gray)

    brightness = summarize_brightness(gray)

    metadata = {
        "timestamp": timestamp,
        "area": round(float(detection["area"]), 2),
        "bbox": detection["bbox"],
        "center": detection["center"],
        "centered": bool(detection["centered"]),
        "aspect_ratio": round(float(detection["aspect_ratio"]), 4),
        "extent": round(float(detection["extent"]), 4),
        "hand_like": bool(detection["hand_like"]),
        "zone": detection["zone"],
        "image": image_path.name,
        "image_relative_path": relative_output_path(image_path, output_root),
        "brightness": brightness,
        "camera_settings": {
            "size": [int(args.size[0]), int(args.size[1])],
            "fps": float(args.fps),
            "frame_duration_us": int(frame_duration_us),
            "exposure_us": int(args.exposure_us),
            "gain": float(args.gain),
            "awbgains": (
                None
                if args.awbgains is None
                else [float(args.awbgains[0]), float(args.awbgains[1])]
            ),
            "brightness": (
                None if args.brightness is None else float(args.brightness)
            ),
            "contrast": None if args.contrast is None else float(args.contrast),
            "saturation": (
                None if args.saturation is None else float(args.saturation)
            ),
            "denoise": None if args.denoise is None else str(args.denoise),
            "quality_min_laplacian_var": float(args.quality_min_laplacian_var),
            "relaxed": bool(args.relaxed),
            "capture_zone_ratio": float(args.capture_zone_ratio),
            "stable_frames": int(args.stable_frames),
            "burst_frames": int(args.burst_frames),
        },
    }
    if burst_summary is not None:
        metadata["burst"] = burst_summary
    if preprocessing_metadata is not None:
        metadata["preprocessing"] = preprocessing_metadata
    if quality_filter_metadata is not None:
        metadata["quality_filter"] = quality_filter_metadata
    meta_path.write_text(json.dumps(metadata, indent=2))
    return image_path, meta_path, metadata


def build_preview(
    gray: np.ndarray,
    mask: np.ndarray,
    detection: Optional[Dict[str, object]],
    stable_count: int,
    stable_frames: int,
    capture_zone_ratio: float,
    capture_armed: bool,
    autostretch: bool,
) -> np.ndarray:
    preview_gray = gray
    if autostretch:
        preview_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    canvas = cv2.cvtColor(preview_gray, cv2.COLOR_GRAY2BGR)
    height, width = gray.shape

    if detection is not None:
        x, y, w, h = detection["bbox"]
        cx, cy = detection["center"]
        zx1, zy1, zx2, zy2 = detection["zone"]
        is_ready = bool(detection["hand_like"] and detection["centered"] and capture_armed)
        if is_ready:
            color = (0, 200, 0)
        elif detection["hand_like"]:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        cv2.rectangle(canvas, (zx1, zy1), (zx2, zy2), (255, 255, 0), 2)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.circle(canvas, (cx, cy), 5, color, -1)
        cv2.putText(
            canvas,
            f"area={int(detection['area'])} ar={detection['aspect_ratio']:.2f} ext={detection['extent']:.2f}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "palm-like" if detection["hand_like"] else "rejected object",
            (x, min(height - 50, y + h + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    else:
        zx1, zy1, zx2, zy2 = center_capture_zone(gray.shape, capture_zone_ratio)
        cv2.rectangle(canvas, (zx1, zy1), (zx2, zy2), (255, 255, 0), 2)

    cv2.putText(
        canvas,
        f"stable={stable_count}/{stable_frames} state={'ARMED' if capture_armed else 'WAIT_REMOVAL'}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    mask_small = cv2.resize(mask, (width // 3, height // 3))
    mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
    canvas[10:10 + mask_small.shape[0], 10:10 + mask_small.shape[1]] = mask_small

    mean_val = float(np.mean(gray))
    cv2.putText(
        canvas,
        f"mean={mean_val:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    argv = sys.argv[1:]
    explicit_options = collect_explicit_options(
        argv,
        {
            "--capture-zone-ratio",
            "--stable-frames",
            "--min-extent",
            "--min-aspect-ratio",
            "--max-aspect-ratio",
            "--preprocess-clahe-clip",
            "--preprocess-clahe-tile",
            "--preprocess-denoise-h",
            "--preprocess-roi-size",
            "--preprocess-center-offset-x",
            "--preprocess-center-offset-y",
            "--preprocess-stretch-percentiles",
            "--preprocess-adaptive-roi",
            "--preprocess-adaptive-roi-scale",
            "--preprocess-palm-core-width-ratio",
        },
    )
    args = parse_args(argv)
    apply_relaxed_preset(args, explicit_options)
    args.blur_kernel = ensure_odd(args.blur_kernel, "blur-kernel")
    args.morph_kernel = ensure_odd(args.morph_kernel, "morph-kernel")
    args.preprocess_vessel_kernel = ensure_odd(
        args.preprocess_vessel_kernel,
        "preprocess-vessel-kernel",
    )

    if not (0.0 < args.min_area_ratio < 1.0):
        raise SystemExit("--min-area-ratio must be between 0 and 1.")
    if not (0.2 <= args.capture_zone_ratio <= 1.0):
        raise SystemExit("--capture-zone-ratio must be between 0.2 and 1.0.")
    if args.min_aspect_ratio <= 0.0:
        raise SystemExit("--min-aspect-ratio must be positive.")
    if args.max_aspect_ratio < args.min_aspect_ratio:
        raise SystemExit("--max-aspect-ratio must be >= --min-aspect-ratio.")
    if not (0.0 < args.min_extent <= 1.0):
        raise SystemExit("--min-extent must be between 0 and 1.")
    if args.stable_frames <= 0:
        raise SystemExit("--stable-frames must be positive.")
    if args.burst_frames <= 0:
        raise SystemExit("--burst-frames must be positive.")
    if args.rearm_empty_frames <= 0:
        raise SystemExit("--rearm-empty-frames must be positive.")
    if not (0.0 <= args.background_update_rate <= 1.0):
        raise SystemExit("--background-update-rate must be between 0 and 1.")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")
    if args.preprocess_roi_size <= 0:
        raise SystemExit("--preprocess-roi-size must be positive.")
    if args.preprocess_final_size <= 0:
        raise SystemExit("--preprocess-final-size must be positive.")
    if args.preprocess_clahe_clip <= 0.0:
        raise SystemExit("--preprocess-clahe-clip must be positive.")
    if args.preprocess_clahe_tile[0] <= 0 or args.preprocess_clahe_tile[1] <= 0:
        raise SystemExit("--preprocess-clahe-tile must use positive dimensions.")
    if args.preprocess_centroid_window < 0:
        raise SystemExit("--preprocess-centroid-window must be >= 0.")
    if args.preprocess_denoise_h is not None and args.preprocess_denoise_h < 0.0:
        raise SystemExit("--preprocess-denoise-h must be >= 0.")
    if not (0.0 < args.preprocess_adaptive_roi_scale <= 1.0):
        raise SystemExit("--preprocess-adaptive-roi-scale must satisfy 0 < scale <= 1.")
    if not (0.0 < args.preprocess_palm_core_width_ratio <= 1.0):
        raise SystemExit(
            "--preprocess-palm-core-width-ratio must satisfy 0 < ratio <= 1."
        )
    if args.quality_min_laplacian_var < 0.0:
        raise SystemExit("--quality-min-laplacian-var must be >= 0.")
    if args.preprocess_stretch_percentiles is not None:
        stretch_low, stretch_high = args.preprocess_stretch_percentiles
        if stretch_low < 0.0 or stretch_high > 100.0 or stretch_low >= stretch_high:
            raise SystemExit(
                "--preprocess-stretch-percentiles must satisfy "
                "0 <= LOW < HIGH <= 100."
            )
    if args.quality_filter and not args.preprocess:
        raise SystemExit("--quality-filter requires --preprocess.")
    if args.save_rejected and not args.quality_filter:
        raise SystemExit("--save-rejected requires --quality-filter.")

    out_dir = Path(args.out_dir)
    preprocessing_config = build_preprocessing_config(args, explicit_options)
    quality_filter_config = build_quality_filter_config(args)
    raw_out_dir, processed_out_dir = resolve_output_dirs(out_dir, args.preprocess)
    rejected_raw_out_dir, rejected_processed_out_dir = resolve_rejected_output_dirs(
        out_dir,
        args.preprocess,
    )
    preview_enabled = args.preview
    preview_warning: Optional[str] = None
    if preview_enabled and not has_gui_display():
        preview_enabled = False
        preview_warning = (
            "Preview disabled: no GUI display detected. Running headless with "
            "console state messages only."
        )

    picam2 = configure_camera(args)

    try:
        background = build_background(
            picam2,
            background_frames=args.background_frames,
            blur_kernel=args.blur_kernel,
        )

        frame_area = args.size[0] * args.size[1]
        min_area = frame_area * args.min_area_ratio
        frame_duration_us = args.frame_duration_us or int(1_000_000 / max(args.fps, 1.0))
        print("Background ready.")
        print(f"Capture size      : {args.size[0]}x{args.size[1]}")
        print(f"Target FPS        : {args.fps:.2f}")
        print(f"Frame duration    : {frame_duration_us} us")
        print(f"Minimum area      : {int(min_area)} px ({args.min_area_ratio:.3f} of frame)")
        if args.awbgains is None:
            print("AWB gains         : auto")
        else:
            print(
                "AWB gains         : "
                f"red={args.awbgains[0]:.3f}, blue={args.awbgains[1]:.3f}"
            )
        if args.brightness is not None:
            print(f"Brightness        : {args.brightness:.3f}")
        if args.contrast is not None:
            print(f"Contrast          : {args.contrast:.3f}")
        if args.saturation is not None:
            print(f"Saturation        : {args.saturation:.3f}")
        if args.denoise is not None:
            print(f"Denoise mode      : {args.denoise}")
        print(f"Stable frames     : {args.stable_frames}")
        print(f"Burst frames      : {args.burst_frames}")
        print(f"Rearm empty frames: {args.rearm_empty_frames}")
        print(f"Cooldown seconds  : {args.cooldown_seconds}")
        print(f"Relaxed mode      : {'enabled' if args.relaxed else 'disabled'}")
        print(f"Preprocessing     : {'enabled' if args.preprocess else 'disabled'}")
        print(f"Quality filter    : {'enabled' if args.quality_filter else 'disabled'}")
        if args.quality_filter:
            print(f"Save rejected     : {'enabled' if args.save_rejected else 'disabled'}")
            print(
                "Quality threshold : "
                f"min_laplacian_var={quality_filter_config.min_laplacian_var}"
            )
        if args.preprocess:
            print(f"Preprocess profile: {preprocessing_config.profile}")
            print(f"Preprocess ROI    : {preprocessing_config.roi_size}x{preprocessing_config.roi_size}")
            print(
                f"Preprocess final  : {preprocessing_config.final_size}x{preprocessing_config.final_size}"
            )
            print(
                "Preprocess CLAHE  : "
                f"clip={preprocessing_config.clahe_clip}, "
                f"tile={preprocessing_config.clahe_tile[0]}x{preprocessing_config.clahe_tile[1]}"
            )
            print(f"Preprocess denoise: h={preprocessing_config.denoise_h}")
            print(
                "Preprocess offset : "
                f"x={preprocessing_config.center_offset_x}, "
                f"y={preprocessing_config.center_offset_y}"
            )
            print(
                "Adaptive ROI      : "
                f"{'enabled' if preprocessing_config.adaptive_roi else 'disabled'}"
            )
            if preprocessing_config.adaptive_roi:
                print(
                    "Adaptive ROI cfg  : "
                    f"scale={preprocessing_config.adaptive_roi_scale}, "
                    f"width_ratio={preprocessing_config.palm_core_width_ratio}"
                )
            if preprocessing_config.stretch_percentiles is None:
                print("Preprocess stretch: min-max")
            else:
                print(
                    "Preprocess stretch: "
                    f"p{preprocessing_config.stretch_percentiles[0]:g}.."
                    f"p{preprocessing_config.stretch_percentiles[1]:g}"
                )
            print(
                "Vessel preview    : "
                f"kernel={preprocessing_config.vessel_preview_kernel}"
            )
        if preview_enabled:
            print("Preview mode      : enabled")
        elif args.preview:
            print("Preview mode      : requested, auto-disabled")
        else:
            print("Preview mode      : disabled")
        if args.preprocess:
            print(f"Output root       : {out_dir.resolve()}")
            print(f"Raw output dir    : {raw_out_dir.resolve()}")
            if processed_out_dir is not None:
                print(f"Processed output  : {processed_out_dir.resolve()}")
        else:
            print(f"Output directory  : {raw_out_dir.resolve()}")
        if preview_warning is not None:
            print(preview_warning)
        print("Press Ctrl+C to stop. If preview is enabled, press q to quit or b to relearn background.")

        stable_count = 0
        last_capture_time = 0.0
        empty_frames = args.rearm_empty_frames
        capture_armed = True
        previous_present = False
        previous_hand_like = False
        previous_ready = False

        while True:
            _, gray = capture_gray_frame(picam2)
            detection, blurred, _, mask = detect_hand(
                gray=gray,
                background=background,
                blur_kernel=args.blur_kernel,
                diff_threshold=args.diff_threshold,
                morph_kernel=args.morph_kernel,
                min_area=min_area,
                capture_zone_ratio=args.capture_zone_ratio,
                min_aspect_ratio=args.min_aspect_ratio,
                max_aspect_ratio=args.max_aspect_ratio,
                min_extent=args.min_extent,
            )

            object_present = detection is not None
            hand_like_present = bool(detection is not None and detection["hand_like"])
            if object_present != previous_present:
                if object_present:
                    if hand_like_present:
                        print("Palm-like object detected.")
                    else:
                        print("Foreground object detected, waiting for palm-like shape.")
                else:
                    print("Object removed.")
                previous_present = object_present
            elif object_present and hand_like_present != previous_hand_like:
                if hand_like_present:
                    print("Palm-like shape confirmed.")
                else:
                    print("Palm-like shape lost, waiting for a better shape.")

            ready_for_capture = bool(
                capture_armed
                and detection is not None
                and detection["hand_like"]
                and detection["centered"]
            )
            if ready_for_capture and not previous_ready:
                print(f"Palm centered, stabilizing for {args.stable_frames} frame(s).")
            elif previous_ready and not ready_for_capture and capture_armed:
                if detection is None:
                    print("Palm readiness lost.")
                elif not detection["hand_like"]:
                    print("Palm readiness lost, shape no longer looks valid.")
                elif not detection["centered"]:
                    print("Palm moved out of the capture zone.")
                else:
                    print("Palm readiness lost.")

            if ready_for_capture:
                stable_count += 1
            else:
                stable_count = 0

            if detection is None:
                empty_frames += 1
                alpha = float(args.background_update_rate)
                if alpha > 0.0:
                    background = cv2.addWeighted(background, 1.0 - alpha, blurred, alpha, 0.0)
            else:
                empty_frames = 0

            if not capture_armed and empty_frames >= args.rearm_empty_frames:
                capture_armed = True
                stable_count = 0
                print("Scanner re-armed.")

            now = time.time()
            if (
                ready_for_capture
                and stable_count >= args.stable_frames
                and (now - last_capture_time) >= args.cooldown_seconds
            ):
                if args.burst_frames > 1:
                    print(f"Capturing burst: {args.burst_frames} frame(s).")
                best_gray, best_detection, burst_summary = capture_burst_best_frame(
                    picam2=picam2,
                    initial_gray=gray,
                    initial_detection=detection,
                    background=background,
                    args=args,
                    min_area=min_area,
                )
                preprocessing_metadata = None
                preprocessing_status: Optional[str] = None
                preprocessing_result = None
                quality_filter_metadata = None
                quality_filter_rejected = False
                if args.preprocess and processed_out_dir is not None:
                    try:
                        preprocessing_result = preprocess_palm_image(
                            best_gray,
                            preprocessing_config,
                        )
                    except Exception as exc:
                        preprocessing_result = None
                        preprocessing_metadata = build_preprocessing_failure_metadata(
                            preprocessing_config,
                            str(exc),
                        )
                        preprocessing_status = f"Preprocessing failed: {exc}"

                if preprocessing_result is not None:
                    quality_filter_metadata = build_quality_filter_metadata(
                        preprocessing_result,
                        args.quality_filter,
                        quality_filter_config,
                    )
                    quality_filter_rejected = bool(
                        args.quality_filter
                        and not quality_filter_metadata["usable"]
                    )
                elif args.quality_filter:
                    quality_filter_metadata = {
                        "enabled": True,
                        "usable": False,
                        "score": 0.0,
                        "reasons": ["preprocessing failed"],
                        "metrics": {},
                        "thresholds": {},
                    }
                    quality_filter_rejected = True

                if quality_filter_rejected and quality_filter_metadata is not None:
                    if args.burst_frames > 1:
                        selected_index = int(burst_summary["selected_frame_index"]) + 1
                        print(
                            f"Selected burst frame {selected_index}/{args.burst_frames} "
                            "for quality filtering."
                        )
                    print(quality_filter_rejection_message(quality_filter_metadata))

                    if args.save_rejected:
                        saved_path, meta_path, metadata = save_capture(
                            rejected_raw_out_dir,
                            out_dir,
                            best_gray,
                            best_detection,
                            args,
                            frame_duration_us,
                            burst_summary,
                            preprocessing_metadata,
                            quality_filter_metadata,
                        )
                        if (
                            preprocessing_result is not None
                            and rejected_processed_out_dir is not None
                        ):
                            try:
                                processed_paths = save_preprocessed_outputs(
                                    rejected_processed_out_dir,
                                    saved_path.stem,
                                    preprocessing_result,
                                )
                                preprocessing_metadata = build_preprocessing_metadata(
                                    out_dir,
                                    processed_paths,
                                    preprocessing_result,
                                    preprocessing_config,
                                )
                                metadata["preprocessing"] = preprocessing_metadata
                                meta_path.write_text(json.dumps(metadata, indent=2))
                            except Exception as exc:
                                preprocessing_metadata = build_preprocessing_failure_metadata(
                                    preprocessing_config,
                                    str(exc),
                                )
                                metadata["preprocessing"] = preprocessing_metadata
                                meta_path.write_text(json.dumps(metadata, indent=2))
                        print(f"Rejected capture saved for debugging: {saved_path}")

                    print("Waiting for object removal before next capture.")
                    last_capture_time = now
                    stable_count = 0
                    empty_frames = 0
                    capture_armed = False
                    previous_hand_like = hand_like_present
                    previous_ready = ready_for_capture
                    continue

                saved_path, meta_path, metadata = save_capture(
                    raw_out_dir,
                    out_dir,
                    best_gray,
                    best_detection,
                    args,
                    frame_duration_us,
                    burst_summary,
                    preprocessing_metadata,
                    quality_filter_metadata if args.quality_filter else None,
                )
                if (
                    args.preprocess
                    and processed_out_dir is not None
                    and preprocessing_status is None
                    and preprocessing_result is not None
                ):
                    try:
                        processed_paths = save_preprocessed_outputs(
                            processed_out_dir,
                            saved_path.stem,
                            preprocessing_result,
                        )
                        preprocessing_metadata = build_preprocessing_metadata(
                            out_dir,
                            processed_paths,
                            preprocessing_result,
                            preprocessing_config,
                        )
                        metadata["preprocessing"] = preprocessing_metadata
                        if args.quality_filter and quality_filter_metadata is not None:
                            metadata["quality_filter"] = quality_filter_metadata
                        meta_path.write_text(json.dumps(metadata, indent=2))
                        preprocessing_status = (
                            "Preprocessed: "
                            f"{processed_paths['roi_path'].name}, "
                            f"{processed_paths['final_path'].name}, "
                            f"{processed_paths['vessel_preview_path'].name}"
                        )
                    except Exception as exc:
                        preprocessing_metadata = build_preprocessing_failure_metadata(
                            preprocessing_config,
                            str(exc),
                        )
                        metadata["preprocessing"] = preprocessing_metadata
                        if args.quality_filter and quality_filter_metadata is not None:
                            metadata["quality_filter"] = quality_filter_metadata
                        meta_path.write_text(json.dumps(metadata, indent=2))
                        preprocessing_status = f"Preprocessing failed: {exc}"
                if args.burst_frames > 1:
                    selected_index = int(burst_summary["selected_frame_index"]) + 1
                    print(
                        f"Selected burst frame {selected_index}/{args.burst_frames} "
                        "for saving."
                    )
                print(f"Captured: {saved_path}")
                if preprocessing_status is not None:
                    print(preprocessing_status)
                print("Waiting for object removal before next capture.")
                last_capture_time = now
                stable_count = 0
                empty_frames = 0
                capture_armed = False

            previous_hand_like = hand_like_present
            previous_ready = ready_for_capture

            if preview_enabled:
                preview = build_preview(
                    gray,
                    mask,
                    detection,
                    stable_count,
                    args.stable_frames,
                    args.capture_zone_ratio,
                    capture_armed,
                    args.preview_autostretch,
                )
                try:
                    cv2.imshow("Hand Detect Capture", preview)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as exc:
                    preview_enabled = False
                    error_message = str(exc).splitlines()[0]
                    print(f"Preview disabled after OpenCV window error: {error_message}")
                    cv2.destroyAllWindows()
                    continue
                if key == ord("q") or key == 27:
                    break
                if key == ord("b"):
                    background = build_background(
                        picam2,
                        background_frames=args.background_frames,
                        blur_kernel=args.blur_kernel,
                    )
                    stable_count = 0
                    empty_frames = args.rearm_empty_frames
                    capture_armed = True
                    print("Background recalibrated.")

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        picam2.stop()
        picam2.close()
        if preview_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

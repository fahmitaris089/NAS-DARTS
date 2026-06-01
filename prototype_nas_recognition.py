"""Prototype palm-vein recognition pipeline using the NAS run6 model.

This script reuses the live capture flow from the scanner prototype, applies
the same adaptive preprocessing used for classes 835/836, and then runs
recognition with a retrained NAS model. It supports two modes:

1. Live Raspberry Pi mode with Picamera2 capture and hand detection.
2. Offline test mode for a single image via ``--test-image``.

Recognition uses a simple reject mechanism:
- reject if preprocessing quality gate says the sample is unusable
- reject if top-class confidence is below threshold
- reject if confidence margin between top-1 and top-2 is too small

Example:
    python3 prototype_nas_recognition.py \
        --model-dir nas_results/retrain_run6_plus2_e100 \
        --subjects 835 836 \
        --preview

    python3 prototype_nas_recognition.py \
        --model-dir nas_results/retrain_run6_plus2_e100 \
        --test-image captures/final_dataset/left/palm_20260510_144934_413065.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch

from genotypes import dict_to_genotype
from model_eval import EvalNetwork
from nas_config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, RETRAIN_CFG
from preprocess_final_dataset_adaptive import preprocess_gray
from utils import get_device


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "nas_results" / "retrain_run6_plus2_e100"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "prototype_results"
DEFAULT_SUBJECTS = ("835", "836")


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


def ensure_odd(value: int, name: str) -> int:
    if value < 3:
        return 3
    if value % 2 == 0:
        value += 1
    return value


def has_gui_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype NAS palm-vein recognition")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--genotype", type=Path, default=None)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--subject-names", nargs="+", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-image", type=Path, default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-autostretch", action="store_true")
    parser.add_argument("--save-rejected", action="store_true")
    parser.add_argument("--reject-threshold", type=float, default=0.75)
    parser.add_argument("--reject-margin", type=float, default=0.15)
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Reject if preprocessing quality gate marks the sample unusable.",
    )
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)

    # Live capture arguments.
    parser.add_argument("--size", type=parse_size, default=(1280, 720))
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--frame-duration-us", type=int, default=0)
    parser.add_argument("--background-frames", type=int, default=30)
    parser.add_argument("--blur-kernel", type=int, default=7)
    parser.add_argument("--diff-threshold", type=int, default=25)
    parser.add_argument("--morph-kernel", type=int, default=7)
    parser.add_argument("--min-area-ratio", type=float, default=0.04)
    parser.add_argument("--capture-zone-ratio", type=float, default=0.60)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.55)
    parser.add_argument("--max-aspect-ratio", type=float, default=1.85)
    parser.add_argument("--min-extent", type=float, default=0.38)
    parser.add_argument("--stable-frames", type=int, default=8)
    parser.add_argument("--burst-frames", type=int, default=3)
    parser.add_argument("--rearm-empty-frames", type=int, default=8)
    parser.add_argument("--background-update-rate", type=float, default=0.02)
    parser.add_argument("--exposure-us", type=int, default=0)
    parser.add_argument("--gain", type=float, default=0.0)
    parser.add_argument("--awbgains", type=parse_awbgains, default=None)
    parser.add_argument("--brightness", type=float, default=None)
    parser.add_argument("--contrast", type=float, default=None)
    parser.add_argument("--saturation", type=float, default=None)
    parser.add_argument("--denoise", type=parse_denoise_mode, default=None)
    parser.add_argument("--relaxed", action="store_true")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_bundle(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model_dir = args.model_dir
    config_path = args.config_path or (model_dir / "config.json")
    model_path = args.model_path or (model_dir / "best_model.pth")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Weights not found: {model_path}")

    cfg = load_json(config_path)
    genotype_path = args.genotype
    if genotype_path is None:
        if "genotype" in cfg:
            genotype = dict_to_genotype(cfg["genotype"])
        else:
            genotype_path = model_dir.parent / "search" / "genotype_final.json"
            genotype = dict_to_genotype(load_json(genotype_path))
    else:
        genotype = dict_to_genotype(load_json(genotype_path))

    subjects = sorted([str(subject) for subject in args.subjects], key=int)
    label_names = args.subject_names or subjects
    if len(label_names) != len(subjects):
        raise ValueError("--subject-names must have the same length as --subjects.")

    model = EvalNetwork(
        genotype=genotype,
        C_init=int(cfg.get("C_init", RETRAIN_CFG["C_init"])),
        num_cells=int(cfg.get("num_cells", RETRAIN_CFG["num_cells"])),
        num_classes=len(subjects),
        auxiliary=False,
        dropout=float(RETRAIN_CFG["dropout"]),
    ).to(device)

    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = {
        key: value for key, value in state_dict.items()
        if not key.startswith("_auxiliary_head")
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return {
        "model": model,
        "config": cfg,
        "subjects": subjects,
        "label_names": label_names,
        "model_dir": model_dir,
        "model_path": model_path,
    }


def preprocess_for_model(image_224: np.ndarray, device: torch.device) -> torch.Tensor:
    image = image_224.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)


@torch.no_grad()
def predict_sample(
    model: torch.nn.Module,
    image_224: np.ndarray,
    device: torch.device,
    subjects: Sequence[str],
    label_names: Sequence[str],
) -> dict[str, Any]:
    logits = model(preprocess_for_model(image_224, device))
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    order = np.argsort(probs)[::-1]
    top_idx = int(order[0])
    second_idx = int(order[1]) if len(order) > 1 else top_idx
    confidence = float(probs[top_idx])
    second_conf = float(probs[second_idx]) if len(order) > 1 else 0.0
    margin = confidence - second_conf

    classes = []
    for idx in order:
        idx = int(idx)
        classes.append(
            {
                "index": idx,
                "subject_id": str(subjects[idx]),
                "label": str(label_names[idx]),
                "probability": float(probs[idx]),
            }
        )

    return {
        "predicted_index": top_idx,
        "predicted_subject": str(subjects[top_idx]),
        "predicted_label": str(label_names[top_idx]),
        "confidence": confidence,
        "second_confidence": second_conf,
        "margin": float(margin),
        "classes": classes,
    }


def decide_recognition(
    prediction: dict[str, Any],
    preprocessing_result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    reasons = []
    quality_filter = preprocessing_result["debug"].get("quality_filter", {})

    if args.quality_filter and not quality_filter.get("usable", True):
        reasons.append("quality_filter")
    if prediction["confidence"] < args.reject_threshold:
        reasons.append("low_confidence")
    if prediction["margin"] < args.reject_margin:
        reasons.append("low_margin")

    return {
        "accepted": len(reasons) == 0,
        "decision": "accepted" if len(reasons) == 0 else "rejected",
        "reasons": reasons,
    }


def resolve_event_dirs(base_out_dir: Path, accepted: bool) -> dict[str, Path]:
    event_root = base_out_dir / ("accepted" if accepted else "rejected")
    return {
        "event_root": event_root,
        "raw_dir": event_root / "raw",
        "processed_dir": event_root / "processed",
        "visualizations_dir": event_root / "visualizations",
        "metadata_dir": event_root / "metadata",
    }


def write_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def save_recognition_event(
    raw_gray: np.ndarray,
    preprocessing_result: dict[str, Any],
    prediction: dict[str, Any],
    decision: dict[str, Any],
    args: argparse.Namespace,
    model_info: dict[str, Any],
    source_image: Optional[Path] = None,
) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dirs = resolve_event_dirs(args.out_dir, decision["accepted"])
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    raw_path = dirs["raw_dir"] / f"recognition_{timestamp}_raw.png"
    roi_path = dirs["processed_dir"] / f"recognition_{timestamp}_roi.png"
    final_path = dirs["processed_dir"] / f"recognition_{timestamp}_final.png"
    clahe_path = dirs["processed_dir"] / f"recognition_{timestamp}_clahe.png"
    vessel_path = dirs["visualizations_dir"] / f"recognition_{timestamp}_vessel.png"
    metadata_path = dirs["metadata_dir"] / f"recognition_{timestamp}.json"

    write_gray(raw_path, raw_gray)
    write_gray(roi_path, preprocessing_result["roi"])
    write_gray(final_path, preprocessing_result["final"])
    write_gray(clahe_path, preprocessing_result["clahe"])
    write_gray(vessel_path, preprocessing_result["vessel_preview"])

    metadata = {
        "timestamp": timestamp,
        "source_image": None if source_image is None else str(source_image),
        "decision": decision,
        "prediction": prediction,
        "model": {
            "model_path": str(model_info["model_path"]),
            "subjects": list(model_info["subjects"]),
            "label_names": list(model_info["label_names"]),
        },
        "thresholds": {
            "reject_threshold": float(args.reject_threshold),
            "reject_margin": float(args.reject_margin),
            "quality_filter": bool(args.quality_filter),
        },
        "preprocessing_debug": {
            "roi_box": list(preprocessing_result["debug"].get("roi_box", ())),
            "rough_center": list(preprocessing_result["debug"].get("rough_center", ())),
            "weighted_center": list(preprocessing_result["debug"].get("weighted_center", ())),
            "final_center": list(preprocessing_result["debug"].get("center_after_offset", ())),
            "hand_bbox": list(preprocessing_result["debug"].get("hand_bbox", ())),
            "palm_bbox": list(preprocessing_result["debug"].get("palm_bbox", ())),
            "quality": preprocessing_result["debug"].get("quality"),
            "quality_filter": preprocessing_result["debug"].get("quality_filter"),
        },
        "saved_outputs": {
            "raw": str(raw_path),
            "roi": str(roi_path),
            "final": str(final_path),
            "clahe": str(clahe_path),
            "vessel_preview": str(vessel_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return final_path, metadata_path


def summarize_result(decision: dict[str, Any], prediction: dict[str, Any]) -> str:
    if decision["accepted"]:
        return (
            f"ACCEPTED -> {prediction['predicted_label']} "
            f"(subject={prediction['predicted_subject']}, "
            f"confidence={prediction['confidence']:.3f}, margin={prediction['margin']:.3f})"
        )
    return (
        f"REJECTED -> best={prediction['predicted_label']} "
        f"(subject={prediction['predicted_subject']}, "
        f"confidence={prediction['confidence']:.3f}, margin={prediction['margin']:.3f}, "
        f"reasons={','.join(decision['reasons'])})"
    )


def annotate_preview(base_preview: np.ndarray, last_summary: Optional[str]) -> np.ndarray:
    if not last_summary:
        return base_preview
    preview = base_preview.copy()
    cv2.putText(
        preview,
        last_summary[:90],
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0) if last_summary.startswith("ACCEPTED") else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def run_single_image(
    image_path: Path,
    model_info: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    preprocessing_result = preprocess_gray(gray)
    prediction = predict_sample(
        model_info["model"],
        preprocessing_result["final"],
        device,
        model_info["subjects"],
        model_info["label_names"],
    )
    decision = decide_recognition(prediction, preprocessing_result, args)
    final_path, metadata_path = save_recognition_event(
        raw_gray=gray,
        preprocessing_result=preprocessing_result,
        prediction=prediction,
        decision=decision,
        args=args,
        model_info=model_info,
        source_image=image_path,
    )
    print(summarize_result(decision, prediction))
    print(f"Saved final image: {final_path}")
    print(f"Saved metadata  : {metadata_path}")


def run_live_recognition(
    args: argparse.Namespace,
    explicit_options: Set[str],
    model_info: dict[str, Any],
    device: torch.device,
) -> None:
    from capture_on_hand_detect import (
        build_background,
        build_preview,
        capture_burst_best_frame,
        capture_gray_frame,
        configure_camera,
        detect_hand,
    )

    apply_relaxed_preset(args, explicit_options)
    args.blur_kernel = ensure_odd(args.blur_kernel, "blur-kernel")
    args.morph_kernel = ensure_odd(args.morph_kernel, "morph-kernel")

    preview_enabled = args.preview and has_gui_display()
    if args.preview and not preview_enabled:
        print("Preview disabled: no GUI display detected.")

    picam2 = configure_camera(args)
    last_capture_time = 0.0
    last_summary: Optional[str] = None

    try:
        background = build_background(
            picam2,
            background_frames=args.background_frames,
            blur_kernel=args.blur_kernel,
        )
        frame_area = args.size[0] * args.size[1]
        min_area = frame_area * args.min_area_ratio
        stable_count = 0
        empty_frames = args.rearm_empty_frames
        capture_armed = True

        print(f"Model dir         : {model_info['model_dir']}")
        print(f"Subjects          : {', '.join(model_info['subjects'])}")
        print(f"Reject threshold  : {args.reject_threshold:.2f}")
        print(f"Reject margin     : {args.reject_margin:.2f}")
        print(f"Quality filter    : {'enabled' if args.quality_filter else 'disabled'}")
        print("Press Ctrl+C to stop. If preview is enabled, press q to quit.")

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

            ready_for_capture = bool(
                capture_armed
                and detection is not None
                and detection["hand_like"]
                and detection["centered"]
            )
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
                best_gray, best_detection, _ = capture_burst_best_frame(
                    picam2=picam2,
                    initial_gray=gray,
                    initial_detection=detection,
                    background=background,
                    args=args,
                    min_area=min_area,
                )
                preprocessing_result = preprocess_gray(best_gray)
                prediction = predict_sample(
                    model_info["model"],
                    preprocessing_result["final"],
                    device,
                    model_info["subjects"],
                    model_info["label_names"],
                )
                decision = decide_recognition(prediction, preprocessing_result, args)
                if decision["accepted"] or args.save_rejected:
                    save_recognition_event(
                        raw_gray=best_gray,
                        preprocessing_result=preprocessing_result,
                        prediction=prediction,
                        decision=decision,
                        args=args,
                        model_info=model_info,
                    )
                last_summary = summarize_result(decision, prediction)
                print(last_summary)
                last_capture_time = now
                stable_count = 0
                empty_frames = 0
                capture_armed = False

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
                preview = annotate_preview(preview, last_summary)
                cv2.imshow("Prototype NAS Recognition", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        picam2.stop()
        picam2.close()
        if preview_enabled:
            cv2.destroyAllWindows()


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
        },
    )
    args = parse_args(argv)
    device = get_device()
    model_info = load_model_bundle(args, device)

    if args.test_image is not None:
        run_single_image(args.test_image, model_info, device, args)
        return

    run_live_recognition(args, explicit_options, model_info, device)


if __name__ == "__main__":
    main()
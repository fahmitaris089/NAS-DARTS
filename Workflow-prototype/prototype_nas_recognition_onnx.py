"""Prototype palm-vein recognition pipeline using ONNX Runtime.

This mirrors the PyTorch prototype pipeline, but uses an exported ONNX model
for inference. Preprocessing and reject logic stay aligned with the existing
adaptive ROI pipeline.

Expected model artifacts in --model-dir:
- model_benchmark.onnx
- model_benchmark_metadata.json

Example:
    python3 prototype_nas_recognition_onnx.py \
        --model-dir nas_results/retrain_run6_plus2_e100 \
        --preview

    python3 prototype_nas_recognition_onnx.py \
        --model-dir nas_results/retrain_run6_plus2_e100 \
        --test-image /path/to/image.png
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from nas_config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE
from preprocess_final_dataset_adaptive import preprocess_gray

try:
    import onnxruntime as ort
except Exception as exc:
    raise SystemExit(f"onnxruntime is required for this script: {exc}") from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "retrain_run6_plus2_e100"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "prototype_results_onnx"


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
    parser = argparse.ArgumentParser(description="Prototype NAS ONNX palm-vein recognition")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--onnx-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--decision-mode", choices=("logits", "verification"), default="logits")
    parser.add_argument("--template-store", type=Path, default=None)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--subject-names", nargs="+", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-image", type=Path, default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-autostretch", action="store_true")
    parser.add_argument("--save-rejected", action="store_true")
    parser.add_argument("--reject-threshold", type=float, default=0.80)
    parser.add_argument("--reject-margin", type=float, default=0.25)
    parser.add_argument("--quality-filter", dest="quality_filter", action="store_true")
    parser.add_argument("--no-quality-filter", dest="quality_filter", action="store_false")
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--consensus-window", type=int, default=5)
    parser.add_argument("--consensus-min-agree", type=int, default=4)
    parser.add_argument("--consensus-min-average-confidence", type=float, default=0.80)
    parser.add_argument("--consensus-min-average-margin", type=float, default=0.25)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--similarity-gap", type=float, default=0.05)

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
    parser.set_defaults(quality_filter=True)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.copy()
    return vector / norm


def resolve_output_name(
    session_output_names: list[str],
    preferred_name: str | None,
    fallback_candidates: Sequence[str],
    fallback_index: int | None,
) -> str | None:
    if preferred_name and preferred_name in session_output_names:
        return preferred_name
    for candidate in fallback_candidates:
        if candidate in session_output_names:
            return candidate
    if fallback_index is None:
        return None
    if 0 <= fallback_index < len(session_output_names):
        return session_output_names[fallback_index]
    return None


def load_template_store(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    templates = payload.get("templates")
    if not isinstance(templates, dict) or not templates:
        raise ValueError(f"Template store is missing templates: {path}")
    return payload


def create_session(model_path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(int(threads), 1)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        options,
        providers=["CPUExecutionProvider"],
    )


def load_backend_bundle(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model_dir
    onnx_path = args.onnx_path or (model_dir / "model_benchmark.onnx")
    metadata_path = args.metadata_path or (model_dir / "model_benchmark_metadata.json")

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"ONNX metadata not found: {metadata_path}")

    metadata = load_json(metadata_path)
    subjects = args.subjects or metadata.get("subjects")
    label_names = args.subject_names or metadata.get("label_names") or subjects
    if subjects is None:
        raise ValueError("Could not infer subjects. Pass --subjects explicitly.")
    if len(label_names) != len(subjects):
        raise ValueError("--subject-names must have the same length as --subjects.")

    session = create_session(onnx_path, args.threads)
    input_name = session.get_inputs()[0].name
    session_output_names = [output.name for output in session.get_outputs()]
    logits_output_name = resolve_output_name(
        session_output_names,
        metadata.get("logits_output_name"),
        ["logits"],
        0,
    )
    embedding_output_name = resolve_output_name(
        session_output_names,
        metadata.get("embedding_output_name"),
        ["embedding", "embeddings", "features"],
        None,
    )
    if logits_output_name is None:
        raise ValueError(f"Could not resolve logits output from ONNX model: {onnx_path}")

    template_store_path = args.template_store
    if template_store_path is None and args.decision_mode == "verification":
        template_store_path = model_dir / "template_store.json"

    template_store = None
    if template_store_path is not None:
        if not template_store_path.exists():
            if args.decision_mode == "verification":
                raise FileNotFoundError(f"Template store not found: {template_store_path}")
        else:
            template_store = load_template_store(template_store_path)

    if args.decision_mode == "verification" and embedding_output_name is None:
        raise ValueError(
            "Verification mode requires an ONNX model that exports embeddings. "
            "Re-export with export_retrain_run6_plus2_onnx.py."
        )

    return {
        "session": session,
        "input_name": input_name,
        "output_names": session_output_names,
        "logits_output_name": logits_output_name,
        "embedding_output_name": embedding_output_name,
        "model_dir": model_dir,
        "onnx_path": onnx_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "subjects": [str(subject) for subject in subjects],
        "label_names": [str(label) for label in label_names],
        "template_store_path": template_store_path,
        "template_store": template_store,
    }


def preprocess_for_model(image_224: np.ndarray) -> np.ndarray:
    image = image_224.astype(np.float32) / 255.0
    rgb = np.stack([image, image, image], axis=0)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    rgb = (rgb - mean) / std
    return np.expand_dims(rgb.astype(np.float32), axis=0)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def run_model_outputs(bundle: dict[str, Any], image_224: np.ndarray) -> dict[str, np.ndarray]:
    outputs = bundle["session"].run(
        bundle["output_names"],
        {bundle["input_name"]: preprocess_for_model(image_224)},
    )
    return {
        name: np.asarray(value, dtype=np.float32)
        for name, value in zip(bundle["output_names"], outputs)
    }


def predict_logits_sample(bundle: dict[str, Any], image_224: np.ndarray) -> dict[str, Any]:
    outputs = run_model_outputs(bundle, image_224)
    logits = outputs[bundle["logits_output_name"]]
    probs = softmax_np(logits)[0]
    order = np.argsort(probs)[::-1]
    top_idx = int(order[0])
    second_idx = int(order[1]) if len(order) > 1 else top_idx
    confidence = float(probs[top_idx])
    second_conf = float(probs[second_idx]) if len(order) > 1 else 0.0

    classes = []
    for idx in order:
        idx = int(idx)
        classes.append(
            {
                "index": idx,
                "subject_id": bundle["subjects"][idx],
                "label": bundle["label_names"][idx],
                "probability": float(probs[idx]),
            }
        )

    return {
        "decision_mode": "logits",
        "score_type": "softmax",
        "predicted_index": top_idx,
        "predicted_subject": bundle["subjects"][top_idx],
        "predicted_label": bundle["label_names"][top_idx],
        "confidence": confidence,
        "second_confidence": second_conf,
        "margin": float(confidence - second_conf),
        "classes": classes,
        "embedding_available": bundle["embedding_output_name"] is not None,
        "embedding_dimension": int(outputs[bundle["embedding_output_name"]].shape[-1])
        if bundle["embedding_output_name"] is not None
        else None,
    }


def predict_verification_sample(bundle: dict[str, Any], image_224: np.ndarray) -> dict[str, Any]:
    if bundle["embedding_output_name"] is None:
        raise ValueError("Verification mode requires an embedding output in the ONNX model.")
    if bundle["template_store"] is None:
        raise ValueError("Verification mode requires a template store.")

    outputs = run_model_outputs(bundle, image_224)
    embedding_batch = outputs[bundle["embedding_output_name"]]
    query_embedding = l2_normalize_vector(embedding_batch[0])

    matches = []
    for subject_id, template_info in bundle["template_store"]["templates"].items():
        template_vector = l2_normalize_vector(np.asarray(template_info["template"], dtype=np.float32))
        similarity = float(np.dot(query_embedding, template_vector))
        matches.append(
            {
                "subject_id": str(subject_id),
                "label": str(template_info.get("label", subject_id)),
                "score": similarity,
            }
        )

    matches.sort(key=lambda item: item["score"], reverse=True)
    best = matches[0]
    second = matches[1] if len(matches) > 1 else best

    return {
        "decision_mode": "verification",
        "score_type": "cosine_similarity",
        "predicted_index": None,
        "predicted_subject": best["subject_id"],
        "predicted_label": best["label"],
        "confidence": float(best["score"]),
        "second_confidence": float(second["score"]),
        "margin": float(best["score"] - second["score"]),
        "classes": matches,
        "embedding_dimension": int(query_embedding.shape[0]),
        "template_store_path": None if bundle["template_store_path"] is None else str(bundle["template_store_path"]),
    }


def predict_sample(bundle: dict[str, Any], image_224: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    if args.decision_mode == "verification":
        return predict_verification_sample(bundle, image_224)
    return predict_logits_sample(bundle, image_224)


def decide_recognition(prediction: dict[str, Any], preprocessing_result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    reasons = []
    quality_filter = preprocessing_result["debug"].get("quality_filter", {})
    score_reason = "low_confidence"
    margin_reason = "low_margin"
    min_score = float(args.reject_threshold)
    min_margin = float(args.reject_margin)

    if prediction.get("decision_mode") == "verification":
        score_reason = "low_similarity"
        margin_reason = "low_similarity_gap"
        min_score = float(args.similarity_threshold)
        min_margin = float(args.similarity_gap)

    if args.quality_filter and not quality_filter.get("usable", True):
        reasons.append("quality_filter")
    if prediction["confidence"] < min_score:
        reasons.append(score_reason)
    if prediction["margin"] < min_margin:
        reasons.append(margin_reason)
    return {
        "accepted": len(reasons) == 0,
        "decision": "accepted" if len(reasons) == 0 else "rejected",
        "reasons": reasons,
    }


def update_consensus_history(
    history: deque[dict[str, Any]],
    prediction: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    history.append(
        {
            "accepted": bool(decision["accepted"]),
            "subject": prediction["predicted_subject"],
            "label": prediction["predicted_label"],
            "confidence": float(prediction["confidence"]),
            "margin": float(prediction["margin"]),
        }
    )


def apply_consensus_gate(
    history: deque[dict[str, Any]],
    prediction: dict[str, Any],
    decision: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not decision["accepted"]:
        return decision

    accepted_entries = [entry for entry in history if entry["accepted"]]
    if len(accepted_entries) < int(args.consensus_min_agree):
        reasons = list(decision["reasons"]) + ["consensus_not_ready"]
        return {
            "accepted": False,
            "decision": "rejected",
            "reasons": reasons,
            "consensus": {
                "window_size": len(history),
                "accepted_votes": len(accepted_entries),
            },
        }

    counts = Counter(entry["subject"] for entry in accepted_entries)
    best_subject, best_votes = counts.most_common(1)[0]
    if best_subject != prediction["predicted_subject"]:
        reasons = list(decision["reasons"]) + ["consensus_subject_mismatch"]
        return {
            "accepted": False,
            "decision": "rejected",
            "reasons": reasons,
            "consensus": {
                "window_size": len(history),
                "accepted_votes": len(accepted_entries),
                "best_subject": best_subject,
                "best_votes": best_votes,
            },
        }

    if best_votes < int(args.consensus_min_agree):
        reasons = list(decision["reasons"]) + ["consensus_too_weak"]
        return {
            "accepted": False,
            "decision": "rejected",
            "reasons": reasons,
            "consensus": {
                "window_size": len(history),
                "accepted_votes": len(accepted_entries),
                "best_subject": best_subject,
                "best_votes": best_votes,
            },
        }

    matching_entries = [entry for entry in accepted_entries if entry["subject"] == best_subject]
    mean_confidence = sum(entry["confidence"] for entry in matching_entries) / len(matching_entries)
    mean_margin = sum(entry["margin"] for entry in matching_entries) / len(matching_entries)

    reasons = list(decision["reasons"])
    if mean_confidence < float(args.consensus_min_average_confidence):
        reasons.append("consensus_low_confidence")
    if mean_margin < float(args.consensus_min_average_margin):
        reasons.append("consensus_low_margin")

    return {
        "accepted": len(reasons) == 0,
        "decision": "accepted" if len(reasons) == 0 else "rejected",
        "reasons": reasons,
        "consensus": {
            "window_size": len(history),
            "accepted_votes": len(accepted_entries),
            "best_subject": best_subject,
            "best_votes": best_votes,
            "mean_confidence": float(mean_confidence),
            "mean_margin": float(mean_margin),
        },
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
    bundle: dict[str, Any],
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
        "backend": "onnxruntime",
        "model": {
            "onnx_path": str(bundle["onnx_path"]),
            "metadata_path": str(bundle["metadata_path"]),
            "subjects": bundle["subjects"],
            "label_names": bundle["label_names"],
            "logits_output_name": bundle.get("logits_output_name"),
            "embedding_output_name": bundle.get("embedding_output_name"),
            "template_store_path": None if bundle.get("template_store_path") is None else str(bundle["template_store_path"]),
        },
        "thresholds": {
            "reject_threshold": float(args.reject_threshold),
            "reject_margin": float(args.reject_margin),
            "similarity_threshold": float(args.similarity_threshold),
            "similarity_gap": float(args.similarity_gap),
            "quality_filter": bool(args.quality_filter),
            "consensus_window": int(args.consensus_window),
            "consensus_min_agree": int(args.consensus_min_agree),
            "consensus_min_average_confidence": float(args.consensus_min_average_confidence),
            "consensus_min_average_margin": float(args.consensus_min_average_margin),
        },
        "decision_mode": args.decision_mode,
        "consensus": decision.get("consensus"),
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
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return final_path, metadata_path


def summarize_result(decision: dict[str, Any], prediction: dict[str, Any]) -> str:
    score_label = "similarity" if prediction.get("decision_mode") == "verification" else "confidence"
    if decision["accepted"]:
        return (
            f"ACCEPTED -> {prediction['predicted_label']} "
            f"(subject={prediction['predicted_subject']}, "
            f"{score_label}={prediction['confidence']:.3f}, margin={prediction['margin']:.3f})"
        )
    return (
        f"REJECTED -> best={prediction['predicted_label']} "
        f"(subject={prediction['predicted_subject']}, "
        f"{score_label}={prediction['confidence']:.3f}, margin={prediction['margin']:.3f}, "
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


def run_single_image(image_path: Path, bundle: dict[str, Any], args: argparse.Namespace) -> None:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    preprocessing_result = preprocess_gray(gray)
    prediction = predict_sample(bundle, preprocessing_result["final"], args)
    decision = decide_recognition(prediction, preprocessing_result, args)
    final_path, metadata_path = save_recognition_event(
        raw_gray=gray,
        preprocessing_result=preprocessing_result,
        prediction=prediction,
        decision=decision,
        args=args,
        bundle=bundle,
        source_image=image_path,
    )
    print(summarize_result(decision, prediction))
    print(f"Saved final image: {final_path}")
    print(f"Saved metadata  : {metadata_path}")


def run_live_recognition(args: argparse.Namespace, explicit_options: Set[str], bundle: dict[str, Any]) -> None:
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
    consensus_history: deque[dict[str, Any]] = deque(maxlen=max(int(args.consensus_window), 1))

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

        print(f"ONNX model        : {bundle['onnx_path']}")
        print(f"Decision mode     : {args.decision_mode}")
        print(f"Subjects          : {', '.join(bundle['subjects'])}")
        if args.decision_mode == "verification":
            print(f"Template store    : {bundle['template_store_path']}")
            print(f"Similarity thr.   : {args.similarity_threshold:.2f}")
            print(f"Similarity gap    : {args.similarity_gap:.2f}")
        else:
            print(f"Reject threshold  : {args.reject_threshold:.2f}")
            print(f"Reject margin     : {args.reject_margin:.2f}")
        print(f"Quality filter    : {'enabled' if args.quality_filter else 'disabled'}")
        print(
            f"Consensus         : {args.consensus_min_agree}/{args.consensus_window} "
            f"avg_conf>={args.consensus_min_average_confidence:.2f} "
            f"avg_margin>={args.consensus_min_average_margin:.2f}"
        )
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
                best_gray, _, _ = capture_burst_best_frame(
                    picam2=picam2,
                    initial_gray=gray,
                    initial_detection=detection,
                    background=background,
                    args=args,
                    min_area=min_area,
                )
                preprocessing_result = preprocess_gray(best_gray)
                prediction = predict_sample(bundle, preprocessing_result["final"], args)
                base_decision = decide_recognition(prediction, preprocessing_result, args)
                update_consensus_history(consensus_history, prediction, base_decision)
                decision = apply_consensus_gate(consensus_history, prediction, base_decision, args)
                if decision["accepted"] or args.save_rejected:
                    save_recognition_event(
                        raw_gray=best_gray,
                        preprocessing_result=preprocessing_result,
                        prediction=prediction,
                        decision=decision,
                        args=args,
                        bundle=bundle,
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
                cv2.imshow("Prototype NAS Recognition ONNX", preview)
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
    bundle = load_backend_bundle(args)

    if args.test_image is not None:
        run_single_image(args.test_image, bundle, args)
        return

    run_live_recognition(args, explicit_options, bundle)


if __name__ == "__main__":
    main()
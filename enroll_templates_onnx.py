"""Enroll subject templates from an embedding-capable ONNX model.

The script reads grayscale images from per-subject folders, preprocesses them
with the same adaptive ROI pipeline as live inference, extracts ONNX
embeddings, then creates one averaged template per subject.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from preprocess_final_dataset_adaptive import preprocess_gray
from prototype_nas_recognition_onnx import (
    create_session,
    l2_normalize_vector,
    load_json,
    preprocess_for_model,
    resolve_output_name,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "nas_results" / "retrain_run6_plus2_e100"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll ONNX embedding templates")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--onnx-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-images", type=int, default=1)
    return parser.parse_args()


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_backend(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model_dir
    onnx_path = args.onnx_path or (model_dir / "model_benchmark.onnx")
    metadata_path = args.metadata_path or (model_dir / "model_benchmark_metadata.json")
    output_path = args.output_path or (model_dir / "template_store.json")

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata = load_json(metadata_path)
    session = create_session(onnx_path, args.threads)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    embedding_output_name = resolve_output_name(
        output_names,
        metadata.get("embedding_output_name"),
        ["embedding", "embeddings", "features"],
        None,
    )
    if embedding_output_name is None:
        raise ValueError(
            "The ONNX model does not expose embeddings. Re-export with "
            "export_retrain_run6_plus2_onnx.py."
        )

    return {
        "model_dir": model_dir,
        "onnx_path": onnx_path,
        "metadata_path": metadata_path,
        "output_path": output_path,
        "metadata": metadata,
        "session": session,
        "input_name": input_name,
        "embedding_output_name": embedding_output_name,
    }


def extract_embedding(bundle: dict[str, Any], image_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    preprocessing_result = preprocess_gray(gray)
    outputs = bundle["session"].run(
        [bundle["embedding_output_name"]],
        {bundle["input_name"]: preprocess_for_model(preprocessing_result["final"])},
    )
    embedding = l2_normalize_vector(np.asarray(outputs[0], dtype=np.float32)[0])
    return embedding, preprocessing_result


def enroll_subject(bundle: dict[str, Any], subject_id: str, folder: Path, min_images: int) -> dict[str, Any]:
    images = image_files(folder)
    if len(images) < min_images:
        raise ValueError(
            f"Subject {subject_id} has only {len(images)} images; require at least {min_images}."
        )

    embeddings = []
    source_images = []
    for image_path in images:
        embedding, _ = extract_embedding(bundle, image_path)
        embeddings.append(embedding)
        source_images.append(str(image_path))

    template = l2_normalize_vector(np.mean(np.stack(embeddings, axis=0), axis=0))
    return {
        "label": subject_id,
        "count": len(embeddings),
        "sources": source_images,
        "template": template.astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    bundle = build_backend(args)
    subjects = args.subjects or bundle["metadata"].get("subjects")
    if not subjects:
        raise ValueError("No subjects specified and none found in model metadata.")

    templates = {}
    for subject in subjects:
        subject_id = str(subject)
        subject_dir = args.input_root / subject_id
        if not subject_dir.exists():
            raise FileNotFoundError(f"Enrollment folder not found for subject {subject_id}: {subject_dir}")
        templates[subject_id] = enroll_subject(bundle, subject_id, subject_dir, args.min_images)

    embedding_dimension = len(next(iter(templates.values()))["template"])
    payload = {
        "created_at": datetime.now().isoformat(),
        "model_dir": str(bundle["model_dir"]),
        "onnx_path": str(bundle["onnx_path"]),
        "metadata_path": str(bundle["metadata_path"]),
        "metric": "cosine_similarity",
        "embedding_dimension": embedding_dimension,
        "subjects": [str(subject) for subject in subjects],
        "templates": templates,
    }

    bundle["output_path"].parent.mkdir(parents=True, exist_ok=True)
    bundle["output_path"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Template store   : {bundle['output_path']}")
    print(f"Subjects         : {', '.join(sorted(templates.keys(), key=int))}")
    print(f"Embedding dim    : {embedding_dimension}")


if __name__ == "__main__":
    main()
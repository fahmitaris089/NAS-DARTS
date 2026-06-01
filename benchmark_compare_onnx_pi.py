"""
Benchmark two ONNX palm-vein classifiers on Raspberry Pi.

Compares:
- accuracy on the shared test split from split_info.json
- per-image latency on CPUExecutionProvider
- model file size

Usage:
  python3 benchmark_compare_onnx_pi.py

  python3 benchmark_compare_onnx_pi.py \
      --data-dir /home/pi/palm-vein/preprocessed_results \
      --split-path /home/pi/palm-vein/split_info.json \
      --model-a /home/pi/palm-vein/MobileNetV3Large/mobilenetv3_benchmark.onnx \
      --model-b /home/pi/palm-vein/run5_efficientNetV2M_t10_a0.5_e500/model_benchmark.onnx
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception as exc:
    raise SystemExit(f"onnxruntime is required: {exc}") from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "preprocessed_results"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "split_info.json"
DEFAULT_MODEL_A = PROJECT_ROOT / "MobileNetV3Large" / "mobilenetv3_benchmark.onnx"
DEFAULT_MODEL_B = (
    PROJECT_ROOT
    / "run5_efficientNetV2M_t10_a0.5_e500"
    / "model_benchmark.onnx"
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass
class Sample:
    path: Path
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two ONNX classifiers on Raspberry Pi")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--model-a", type=Path, default=DEFAULT_MODEL_A)
    parser.add_argument("--model-b", type=Path, default=DEFAULT_MODEL_B)
    parser.add_argument("--label-a", type=str, default="MobileNetV3Large")
    parser.add_argument("--label-b", type=str, default="KD-EfficientNetV2M")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = use full test split")
    parser.add_argument("--save-path", type=Path, default=PROJECT_ROOT / "benchmark_compare_onnx_pi_results.json")
    return parser.parse_args()


def load_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_label_map(subjects: list[str]) -> dict[str, int]:
    ordered = sorted((str(subject) for subject in subjects), key=int)
    return {subject: idx for idx, subject in enumerate(ordered)}


def build_test_samples(data_dir: Path, split: dict[str, Any], max_samples: int) -> list[Sample]:
    label_map = build_label_map(split["subjects"])
    samples: list[Sample] = []
    for subject_id, filename in split["test"]:
        image_path = data_dir / str(subject_id) / filename
        if image_path.exists():
            samples.append(Sample(path=image_path, label=label_map[str(subject_id)]))
    if not samples:
        raise FileNotFoundError(f"No test samples found under {data_dir}")
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def preprocess_image(path: Path, input_size: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize((input_size, input_size), Image.BILINEAR)
    gray = np.asarray(image, dtype=np.float32) / 255.0
    rgb = np.stack([gray, gray, gray], axis=0)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(rgb.astype(np.float32), axis=0)


def create_session(model_path: Path, threads: int) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(int(threads), 1)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


def benchmark_model(
    model_path: Path,
    samples: list[Sample],
    input_size: int,
    threads: int,
    warmup: int,
) -> dict[str, Any]:
    session = create_session(model_path, threads)
    input_name = session.get_inputs()[0].name

    cached_inputs = [preprocess_image(sample.path, input_size) for sample in samples]
    dummy = cached_inputs[0]

    for _ in range(max(warmup, 0)):
        session.run(None, {input_name: dummy})

    latencies_ms: list[float] = []
    correct = 0

    for sample, array in zip(samples, cached_inputs):
        start = time.perf_counter()
        outputs = session.run(None, {input_name: array})
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)

        logits = np.asarray(outputs[0], dtype=np.float32)
        pred = int(np.argmax(logits[0]))
        if pred == sample.label:
            correct += 1

    latency = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "model_path": str(model_path),
        "num_samples": len(samples),
        "accuracy": correct / len(samples),
        "correct": correct,
        "file_size_mb": model_path.stat().st_size / 1e6,
        "latency_mean_ms": float(latency.mean()),
        "latency_median_ms": float(np.median(latency)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "latency_std_ms": float(latency.std()),
        "threads": threads,
    }


def print_result(label: str, result: dict[str, Any]) -> None:
    print(f"\n{label}")
    print(f"  model     : {result['model_path']}")
    print(f"  accuracy  : {result['accuracy'] * 100:.2f}% ({result['correct']}/{result['num_samples']})")
    print(f"  size      : {result['file_size_mb']:.3f} MB")
    print(
        "  latency   : "
        f"mean={result['latency_mean_ms']:.2f} ms  "
        f"median={result['latency_median_ms']:.2f} ms  "
        f"p95={result['latency_p95_ms']:.2f} ms"
    )


def main() -> None:
    args = parse_args()
    split = load_split(args.split_path)
    samples = build_test_samples(args.data_dir, split, args.max_samples)

    print("Benchmark ONNX on Raspberry Pi")
    print(f"  data dir   : {args.data_dir}")
    print(f"  split path : {args.split_path}")
    print(f"  test imgs  : {len(samples)}")
    print(f"  threads    : {args.threads}")

    result_a = benchmark_model(args.model_a, samples, args.input_size, args.threads, args.warmup)
    result_b = benchmark_model(args.model_b, samples, args.input_size, args.threads, args.warmup)

    print_result(args.label_a, result_a)
    print_result(args.label_b, result_b)

    faster = args.label_a if result_a["latency_mean_ms"] < result_b["latency_mean_ms"] else args.label_b
    more_accurate = args.label_a if result_a["accuracy"] > result_b["accuracy"] else args.label_b

    summary = {
        "data_dir": str(args.data_dir),
        "split_path": str(args.split_path),
        "input_size": args.input_size,
        "threads": args.threads,
        "warmup": args.warmup,
        "num_samples": len(samples),
        "model_a": {"label": args.label_a, **result_a},
        "model_b": {"label": args.label_b, **result_b},
        "summary": {
            "faster_model": faster,
            "more_accurate_model": more_accurate,
            "latency_speedup_x": (
                result_b["latency_mean_ms"] / result_a["latency_mean_ms"]
                if result_a["latency_mean_ms"] < result_b["latency_mean_ms"]
                else result_a["latency_mean_ms"] / result_b["latency_mean_ms"]
            ),
            "accuracy_gap_pct_points": abs(result_a["accuracy"] - result_b["accuracy"]) * 100.0,
        },
    }

    args.save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults saved: {args.save_path}")


if __name__ == "__main__":
    main()

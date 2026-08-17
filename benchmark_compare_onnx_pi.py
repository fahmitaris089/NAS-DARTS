"""
Benchmark two ONNX palm-vein classifiers.

Flexible: supports comparing models with different numbers of output classes
(e.g. 834-class vs 2-class) on the same image set. Each model uses its own
subject-to-label mapping. Supports flat and multi-distance nested data dirs.

--- Usage examples ---

# 1. Default: NAS-C4 (2-class) vs MobileNetV3 baseline (2-class), local dataset
  python3 benchmark_compare_onnx_pi.py \
      --save-path benchmark_compare_onnx_pi_results.json

# 2. 834-class MobileNetV3 vs 2-class NAS-C4 on private 2-class test set
#    The 834-class model's subjects 835/836 map to their SCUT class indices
  python3 benchmark_compare_onnx_pi.py \
      --model-a MobileNetV3Large/mobilenetv3_benchmark.onnx \
      --label-a "MobileNetV3L-834cls" \
      --subject-map-a 835:834 836:835 \
      --model-b nas_results/retrain_mobile_v2_C4/model_benchmark.onnx \
      --label-b "NAS-DARTS-C4" \
      --save-path benchmark_compare_onnx_pi_results.json

# 3. Latency-only (no accuracy), e.g. when subjects not in 834-class training set
  python3 benchmark_compare_onnx_pi.py \
      --model-a MobileNetV3Large/mobilenetv3_benchmark.onnx \
      --label-a "MobileNetV3L-834cls" \
      --skip-accuracy-a \
      --save-path benchmark_compare_onnx_pi_results.json

# 4. Pi 5 (copy dataset_multi_distance ke Pi 5, struktur nested tetap didukung)
  python3 benchmark_compare_onnx_pi.py \
      --data-dir /home/pi/NAS-DARTS/dataset_multi_distance \
      --split-path /home/pi/NAS-DARTS/nas_results/retrain_mobile_v2_C4/split_info_converted.json \
      --model-a /home/pi/NAS-DARTS/nas_results/retrain_mobile_v2_C4/model_benchmark.onnx \
      --model-b /home/pi/NAS-DARTS/nas_results/baseline_mobilenetv3/mobilenetv3_benchmark.onnx \
      --threads 4 \
      --save-path /home/pi/NAS-DARTS/benchmark_compare_onnx_pi_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from palm_input_preprocessing import (
    INPUT_PROFILES,
    input_profile_metadata,
    preprocess_path_to_imagenet_bchw,
)

try:
    import onnxruntime as ort
except Exception as exc:
    raise SystemExit(f"onnxruntime is required: {exc}") from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset_multi_distance"
DEFAULT_SPLIT_PATH = (
    PROJECT_ROOT / "nas_results" / "retrain_mobile_v2_C4" / "split_info_converted.json"
)
DEFAULT_MODEL_A = (
    PROJECT_ROOT / "nas_results" / "retrain_mobile_v2_C4" / "model_benchmark.onnx"
)
DEFAULT_MODEL_B = (
    PROJECT_ROOT / "nas_results" / "baseline_mobilenetv3" / "mobilenetv3_benchmark.onnx"
)

@dataclass
class Sample:
    path: Path
    subject_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two ONNX classifiers")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--model-a", type=Path, default=DEFAULT_MODEL_A)
    parser.add_argument("--model-b", type=Path, default=DEFAULT_MODEL_B)
    parser.add_argument("--label-a", type=str, default="NAS-DARTS-C4")
    parser.add_argument("--label-b", type=str, default="MobileNetV3Large-baseline")
    parser.add_argument(
        "--subject-map-a", nargs="+", default=None, metavar="SUBJ:IDX",
        help="Per-model subject→label mapping for model-a, e.g. 835:0 836:1. "
             "Default: auto-infer from split (sorted order).",
    )
    parser.add_argument(
        "--subject-map-b", nargs="+", default=None, metavar="SUBJ:IDX",
        help="Same as --subject-map-a but for model-b.",
    )
    parser.add_argument(
        "--skip-accuracy-a", action="store_true",
        help="Skip accuracy evaluation for model-a (latency-only).",
    )
    parser.add_argument(
        "--skip-accuracy-b", action="store_true",
        help="Skip accuracy evaluation for model-b (latency-only).",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--input-profile-a", choices=INPUT_PROFILES, default="legacy")
    parser.add_argument("--input-profile-b", choices=INPUT_PROFILES, default="legacy")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--iterations", type=int, default=0,
        help="Timed iterations per model; 0 uses the number of test samples",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 = use full test split")
    parser.add_argument("--save-path", type=Path,
                        default=PROJECT_ROOT / "benchmark_compare_onnx_pi_results.json")
    return parser.parse_args()


def load_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_label_map(subjects: list[str]) -> dict[str, int]:
    ordered = sorted((str(subject) for subject in subjects), key=int)
    return {subject: idx for idx, subject in enumerate(ordered)}


def parse_subject_map(pairs: list[str] | None) -> dict[str, int] | None:
    """Parse ['835:0', '836:1'] → {'835': 0, '836': 1}, or None if not provided."""
    if pairs is None:
        return None
    result: dict[str, int] = {}
    for pair in pairs:
        subject, idx = pair.split(":")
        result[subject.strip()] = int(idx.strip())
    return result


def build_test_samples(data_dir: Path, split: dict[str, Any], max_samples: int) -> list[Sample]:
    samples: list[Sample] = []
    missing: list[str] = []

    for subject_id, filename in split["test"]:
        flat_path = data_dir / str(subject_id) / filename
        if flat_path.exists():
            samples.append(Sample(path=flat_path, subject_id=str(subject_id)))
        else:
            subject_dir = data_dir / str(subject_id)
            found = list(subject_dir.rglob(filename)) if subject_dir.exists() else []
            if found:
                samples.append(Sample(path=found[0], subject_id=str(subject_id)))
            else:
                missing.append(f"{subject_id}/{filename}")

    if missing:
        print(f"  [warn] {len(missing)} test files not found (skipped)")
    if not samples:
        raise FileNotFoundError(f"No test samples found under {data_dir}")
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def preprocess_image(path: Path, input_size: int, input_profile: str) -> np.ndarray:
    return preprocess_path_to_imagenet_bchw(
        str(path), input_size=input_size, profile=input_profile,
    )


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
    input_profile: str,
    iterations: int,
    label_map: dict[str, int] | None = None,
    skip_accuracy: bool = False,
) -> dict[str, Any]:
    session = create_session(model_path, threads)
    input_name = session.get_inputs()[0].name

    cached_inputs = [
        preprocess_image(sample.path, input_size, input_profile) for sample in samples
    ]
    dummy = cached_inputs[0]

    for _ in range(max(warmup, 0)):
        session.run(None, {input_name: dummy})

    latencies_ms: list[float] = []
    preprocessing_ms: list[float] = []
    end_to_end_ms: list[float] = []
    correct = 0
    evaluated = 0

    if not skip_accuracy and label_map is not None:
        for sample, array in zip(samples, cached_inputs):
            outputs = session.run(None, {input_name: array})
            if sample.subject_id in label_map:
                logits = np.asarray(outputs[0], dtype=np.float32)
                pred = int(np.argmax(logits[0]))
                if pred == label_map[sample.subject_id]:
                    correct += 1
                evaluated += 1

    timed_iterations = iterations if iterations > 0 else len(samples)
    for index in range(timed_iterations):
        sample = samples[index % len(samples)]
        array = cached_inputs[index % len(cached_inputs)]
        start = time.perf_counter()
        session.run(None, {input_name: array})
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)

        preprocess_start = time.perf_counter()
        live_array = preprocess_image(sample.path, input_size, input_profile)
        inference_start = time.perf_counter()
        session.run(None, {input_name: live_array})
        e2e_end = time.perf_counter()
        preprocessing_ms.append((inference_start - preprocess_start) * 1000.0)
        end_to_end_ms.append((e2e_end - preprocess_start) * 1000.0)

    latency = np.asarray(latencies_ms, dtype=np.float64)
    preprocessing = np.asarray(preprocessing_ms, dtype=np.float64)
    end_to_end = np.asarray(end_to_end_ms, dtype=np.float64)
    accuracy = (correct / evaluated) if evaluated > 0 else None
    return {
        "model_path": str(model_path),
        "num_samples": len(samples),
        "timed_iterations": timed_iterations,
        "accuracy": accuracy,
        "correct": correct if evaluated > 0 else None,
        "accuracy_evaluated": evaluated,
        "file_size_mb": model_path.stat().st_size / 1e6,
        "input_profile": input_profile,
        "input_profile_metadata": input_profile_metadata(input_profile),
        "latency_scope": "model_only_preprocessed_input",
        "latency_mean_ms": float(latency.mean()),
        "latency_median_ms": float(np.median(latency)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "latency_std_ms": float(latency.std()),
        "preprocessing_mean_ms": float(preprocessing.mean()),
        "preprocessing_median_ms": float(np.median(preprocessing)),
        "preprocessing_p95_ms": float(np.percentile(preprocessing, 95)),
        "end_to_end_mean_ms": float(end_to_end.mean()),
        "end_to_end_median_ms": float(np.median(end_to_end)),
        "end_to_end_p95_ms": float(np.percentile(end_to_end, 95)),
        "threads": threads,
    }


def print_result(label: str, result: dict[str, Any]) -> None:
    print(f"\n{label}")
    print(f"  model     : {result['model_path']}")
    if result["accuracy"] is not None:
        print(f"  accuracy  : {result['accuracy'] * 100:.2f}% ({result['correct']}/{result['accuracy_evaluated']})")
    else:
        print("  accuracy  : N/A (skipped or label map not provided)")
    print(f"  size      : {result['file_size_mb']:.3f} MB")
    print(
        "  model-only: "
        f"mean={result['latency_mean_ms']:.2f} ms  "
        f"median={result['latency_median_ms']:.2f} ms  "
        f"p95={result['latency_p95_ms']:.2f} ms"
    )
    print(
        "  end-to-end: "
        f"mean={result['end_to_end_mean_ms']:.2f} ms  "
        f"median={result['end_to_end_median_ms']:.2f} ms  "
        f"p95={result['end_to_end_p95_ms']:.2f} ms"
    )
    print(f"  input     : {result['input_profile']}")


def main() -> None:
    args = parse_args()
    split = load_split(args.split_path)
    samples = build_test_samples(args.data_dir, split, args.max_samples)

    # Default label map: sorted subjects → 0, 1, ... (from split)
    default_label_map = build_label_map(split["subjects"])
    label_map_a = parse_subject_map(args.subject_map_a) or default_label_map
    label_map_b = parse_subject_map(args.subject_map_b) or default_label_map

    print("Benchmark ONNX")
    print(f"  data dir   : {args.data_dir}")
    print(f"  split path : {args.split_path}")
    print(f"  test imgs  : {len(samples)}")
    print(f"  threads    : {args.threads}")
    print(f"  label map A: {label_map_a}" + (" [skip accuracy]" if args.skip_accuracy_a else ""))
    print(f"  label map B: {label_map_b}" + (" [skip accuracy]" if args.skip_accuracy_b else ""))

    result_a = benchmark_model(
        args.model_a, samples, args.input_size, args.threads, args.warmup,
        args.input_profile_a, args.iterations,
        label_map=label_map_a, skip_accuracy=args.skip_accuracy_a,
    )
    result_b = benchmark_model(
        args.model_b, samples, args.input_size, args.threads, args.warmup,
        args.input_profile_b, args.iterations,
        label_map=label_map_b, skip_accuracy=args.skip_accuracy_b,
    )

    print_result(args.label_a, result_a)
    print_result(args.label_b, result_b)

    faster = args.label_a if result_a["latency_mean_ms"] < result_b["latency_mean_ms"] else args.label_b
    acc_a, acc_b = result_a["accuracy"], result_b["accuracy"]
    more_accurate = (
        args.label_a if (acc_a or 0) > (acc_b or 0) else args.label_b
        if acc_a is not None or acc_b is not None else "N/A"
    )

    summary = {
        "data_dir": str(args.data_dir),
        "split_path": str(args.split_path),
        "input_size": args.input_size,
        "threads": args.threads,
        "warmup": args.warmup,
        "timed_iterations": args.iterations or len(samples),
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
            "accuracy_gap_pct_points": (
                abs((acc_a or 0) - (acc_b or 0)) * 100.0
                if acc_a is not None and acc_b is not None else None
            ),
        },
    }

    args.save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults saved: {args.save_path}")


if __name__ == "__main__":
    main()

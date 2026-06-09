"""
Quantize both ONNX models to INT8 (static) then benchmark FP32 vs INT8
for NAS-DARTS C4 and MobileNetV3Large on palm-vein dataset.

Usage (Mac):
  python3 benchmark_int8_compare.py

Usage (Pi 5):
  python3 benchmark_int8_compare.py \
      --data-dir dataset_multi_distance \
      --split-path nas_results/retrain_mobile_v2_C4/split_info_converted.json \
      --model-a nas_results/retrain_mobile_v2_C4/model_benchmark.onnx \
      --model-b nas_results/baseline_mobilenetv3/mobilenetv3_benchmark.onnx \
      --threads 4 \
      --save-path benchmark_int8_compare_results.json

    python3 benchmark_compare_onnx_pi.py     
    --model-a nas_results/retrain_mobile_v2_C4/model_benchmark_int8.onnx     
    --label-a "NAS-DARTS-C4-INT8"     
    --model-b nas_results/baseline_mobilenetv3/mobilenetv3_benchmark.onnx     
    --label-b "MobileNetV3Large-FP32"     
    --threads 4     
    --save-path benchmark_int8_vs_fp32_pi_results.json

"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
except Exception as exc:
    raise SystemExit(f"onnxruntime[quantization] required: {exc}") from exc


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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="INT8 quantization + compare benchmark")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    p.add_argument("--model-a", type=Path, default=DEFAULT_MODEL_A)
    p.add_argument("--model-b", type=Path, default=DEFAULT_MODEL_B)
    p.add_argument("--label-a", type=str, default="NAS-DARTS-C4")
    p.add_argument("--label-b", type=str, default="MobileNetV3Large-baseline")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--num-calib", type=int, default=100,
                   help="Max calibration images (uses non-test images from data-dir)")
    p.add_argument("--save-path", type=Path,
                   default=PROJECT_ROOT / "benchmark_int8_compare_results.json")
    return p.parse_args()


# ── preprocessing ────────────────────────────────────────────────────────────

def preprocess(path: Path, input_size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    rgb = np.stack([arr, arr, arr], axis=0)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(rgb.astype(np.float32), axis=0)


# ── split loading ─────────────────────────────────────────────────────────────

def load_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_file(data_dir: Path, subject_id: str, filename: str) -> Path | None:
    flat = data_dir / str(subject_id) / filename
    if flat.exists():
        return flat
    subject_dir = data_dir / str(subject_id)
    if subject_dir.exists():
        found = list(subject_dir.rglob(filename))
        if found:
            return found[0]
    return None


def build_label_map(subjects: list) -> dict[str, int]:
    ordered = sorted((str(s) for s in subjects), key=int)
    return {s: i for i, s in enumerate(ordered)}


def collect_test_samples(data_dir: Path, split: dict, label_map: dict[str, int]) -> list[tuple[Path, int]]:
    samples = []
    for subject_id, filename in split["test"]:
        p = find_file(data_dir, str(subject_id), filename)
        if p is not None and str(subject_id) in label_map:
            samples.append((p, label_map[str(subject_id)]))
    if not samples:
        raise FileNotFoundError(f"No test samples found under {data_dir}")
    return samples


def collect_calib_images(data_dir: Path, split: dict, num_calib: int) -> list[Path]:
    """Use non-test images for calibration to avoid leakage."""
    test_set = {filename for _, filename in split["test"]}
    images = [
        p for p in data_dir.rglob("*.bmp")
        if p.name not in test_set
    ]
    images.sort()
    if not images:
        # Fall back to all images if no non-test images available
        images = sorted(data_dir.rglob("*.bmp"))
    return images[:num_calib]


# ── calibration reader ────────────────────────────────────────────────────────

class CalibReader(CalibrationDataReader):
    def __init__(self, paths: list[Path], input_name: str, input_size: int):
        self._data = [preprocess(p, input_size) for p in paths]
        self._input_name = input_name
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self._data):
            return None
        item = {self._input_name: self._data[self._idx]}
        self._idx += 1
        return item

    def rewind(self):
        self._idx = 0


# ── ONNX session ──────────────────────────────────────────────────────────────

def make_session(model_path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(int(threads), 1)
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])


# ── quantization ──────────────────────────────────────────────────────────────

def quantize_model(fp32_path: Path, calib_images: list[Path], input_size: int) -> Path:
    int8_path = fp32_path.with_suffix("").parent / (fp32_path.stem + "_int8.onnx")
    if int8_path.exists():
        print(f"  [skip] INT8 model already exists: {int8_path.name}")
        return int8_path

    print(f"  Quantizing {fp32_path.name} → {int8_path.name} ...")
    sess_tmp = make_session(fp32_path, 1)
    input_name = sess_tmp.get_inputs()[0].name
    reader = CalibReader(calib_images, input_name, input_size)

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"  Done. Size: {int8_path.stat().st_size / 1e6:.3f} MB")
    return int8_path


# ── benchmark ─────────────────────────────────────────────────────────────────

def benchmark(
    model_path: Path,
    test_samples: list[tuple[Path, int]],
    input_size: int,
    threads: int,
    warmup: int,
) -> dict[str, Any]:
    sess = make_session(model_path, threads)
    input_name = sess.get_inputs()[0].name

    # Cache all inputs
    cached = [(preprocess(p, input_size), label) for p, label in test_samples]
    dummy = cached[0][0]

    for _ in range(max(warmup, 0)):
        sess.run(None, {input_name: dummy})

    latencies_ms: list[float] = []
    correct = 0

    for arr, label in cached:
        t0 = time.perf_counter()
        out = sess.run(None, {input_name: arr})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        pred = int(np.argmax(np.asarray(out[0], dtype=np.float32)[0]))
        if pred == label:
            correct += 1

    lat = np.asarray(latencies_ms, dtype=np.float64)
    n = len(test_samples)
    return {
        "model_path": str(model_path),
        "size_mb": round(model_path.stat().st_size / 1e6, 4),
        "accuracy": round(correct / n, 4) if n > 0 else None,
        "correct": correct,
        "num_samples": n,
        "mean_ms": round(float(lat.mean()), 3),
        "median_ms": round(float(np.median(lat)), 3),
        "p95_ms": round(float(np.percentile(lat, 95)), 3),
        "std_ms": round(float(lat.std()), 3),
    }


# ── print helpers ─────────────────────────────────────────────────────────────

def print_row(label: str, r: dict) -> None:
    acc_str = f"{r['accuracy'] * 100:.1f}% ({r['correct']}/{r['num_samples']})" if r["accuracy"] is not None else "N/A"
    print(f"  {label:<30}  acc={acc_str:<14}  size={r['size_mb']:.3f} MB  "
          f"mean={r['mean_ms']:.2f}ms  median={r['median_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    split = load_split(args.split_path)
    label_map = build_label_map(split["subjects"])
    test_samples = collect_test_samples(args.data_dir, split, label_map)
    calib_images = collect_calib_images(args.data_dir, split, args.num_calib)

    print("=" * 70)
    print("  INT8 Quantization + Benchmark")
    print("=" * 70)
    print(f"  data dir      : {args.data_dir}")
    print(f"  test samples  : {len(test_samples)}")
    print(f"  calib images  : {len(calib_images)}")
    print(f"  threads       : {args.threads}")
    print(f"  warmup        : {args.warmup}")

    # ── Quantize both models ──────────────────────────────────────────────────
    print("\n[1/2] Quantizing models ...")
    int8_a = quantize_model(args.model_a, calib_images, args.input_size)
    int8_b = quantize_model(args.model_b, calib_images, args.input_size)

    # ── Benchmark all 4 variants ──────────────────────────────────────────────
    print("\n[2/2] Benchmarking (warmup={}) ...".format(args.warmup))
    r_fp32_a = benchmark(args.model_a, test_samples, args.input_size, args.threads, args.warmup)
    r_int8_a = benchmark(int8_a,       test_samples, args.input_size, args.threads, args.warmup)
    r_fp32_b = benchmark(args.model_b, test_samples, args.input_size, args.threads, args.warmup)
    r_int8_b = benchmark(int8_b,       test_samples, args.input_size, args.threads, args.warmup)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Results")
    print("=" * 70)
    print_row(f"{args.label_a} FP32", r_fp32_a)
    print_row(f"{args.label_a} INT8", r_int8_a)
    print("-" * 70)
    print_row(f"{args.label_b} FP32", r_fp32_b)
    print_row(f"{args.label_b} INT8", r_int8_b)

    speedup_a = r_fp32_a["mean_ms"] / r_int8_a["mean_ms"] if r_int8_a["mean_ms"] > 0 else float("nan")
    speedup_b = r_fp32_b["mean_ms"] / r_int8_b["mean_ms"] if r_int8_b["mean_ms"] > 0 else float("nan")
    print(f"\n  INT8 speedup {args.label_a}: {speedup_a:.2f}x")
    print(f"  INT8 speedup {args.label_b}: {speedup_b:.2f}x")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {
        "data_dir": str(args.data_dir),
        "split_path": str(args.split_path),
        "threads": args.threads,
        "warmup": args.warmup,
        "num_calib": len(calib_images),
        "model_a": {
            "label": args.label_a,
            "fp32": r_fp32_a,
            "int8": r_int8_a,
            "int8_path": str(int8_a),
            "speedup_x": round(speedup_a, 3),
        },
        "model_b": {
            "label": args.label_b,
            "fp32": r_fp32_b,
            "int8": r_int8_b,
            "int8_path": str(int8_b),
            "speedup_x": round(speedup_b, 3),
        },
    }
    args.save_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results saved : {args.save_path}")


if __name__ == "__main__":
    main()

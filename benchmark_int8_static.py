"""
Static INT8 quantization + benchmark for P-DARTS ONNX model.

Usage:
  source .venv/bin/activate
  python3 benchmark_int8_static.py \
      --model_dir knowledge_distilation/kd_results/run5_efficientNetV2M_t10_a0.5_e500 \
      --calib_dir preprocessed_results \
      --num_calib 200
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

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
except Exception as e:
    raise SystemExit(f"onnxruntime quantization modules unavailable: {e}")


ROOT = Path(__file__).resolve().parent


def print_section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def preprocess_bmp(path: Path, input_size: int = 224) -> np.ndarray:
    """BMP grayscale -> normalized RGB tensor BCHW float32."""
    img = Image.open(path).convert("L").resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # GrayscaleToRGB + ImageNet normalization (same as training/eval pipeline)
    rgb = np.stack([arr, arr, arr], axis=0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb = (rgb - mean) / std

    return np.expand_dims(rgb.astype(np.float32), axis=0)


def collect_calibration_images(calib_dir: Path, limit: int) -> List[Path]:
    images = sorted(calib_dir.rglob("*.bmp"))
    if not images:
        raise FileNotFoundError(f"No .bmp files found under {calib_dir}")
    return images[: min(limit, len(images))]


class PalmVeinCalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths: List[Path], input_name: str, input_size: int):
        self.input_name = input_name
        self.input_size = input_size
        self._data = [preprocess_bmp(p, input_size) for p in image_paths]
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self._data):
            return None
        x = self._data[self._idx]
        self._idx += 1
        return {self.input_name: x}


def make_session(model_path: Path, threads: int):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


def benchmark_onnx(model_path: Path, input_size: int, threads: int, warmup: int, runs: int):
    sess = make_session(model_path, threads)
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    arr = np.array(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
    }


def main():
    parser = argparse.ArgumentParser(description="Static INT8 quantization benchmark")
    parser.add_argument(
        "--model_dir",
        default="knowledge_distilation/kd_results/run5_efficientNetV2M_t10_a0.5_e500",
        help="Folder containing model_benchmark.onnx",
    )
    parser.add_argument("--calib_dir", default="preprocessed_results", help="Calibration image root")
    parser.add_argument("--num_calib", type=int, default=200, help="Number of calibration images")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    model_dir = ROOT / args.model_dir
    fp32_path = model_dir / "model_benchmark.onnx"
    int8_path = model_dir / "model_benchmark_int8_static.onnx"
    calib_dir = ROOT / args.calib_dir

    if not fp32_path.exists():
        raise FileNotFoundError(f"FP32 ONNX not found: {fp32_path}")
    if not calib_dir.exists():
        raise FileNotFoundError(f"Calibration dir not found: {calib_dir}")

    print_section("STATIC INT8 QUANTIZATION")
    print(f"  FP32 model     : {fp32_path}")
    print(f"  INT8 output    : {int8_path}")
    print(f"  Calibration dir: {calib_dir}")

    calib_images = collect_calibration_images(calib_dir, args.num_calib)
    print(f"  Calibration img: {len(calib_images)} bmp")

    fp32_sess = make_session(fp32_path, args.threads)
    input_name = fp32_sess.get_inputs()[0].name
    reader = PalmVeinCalibrationReader(calib_images, input_name, args.input_size)

    # Try per-channel first (requires opset >= 13), fallback to per-tensor
    try:
        quantize_static(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
    except ValueError as e:
        if "opset" in str(e).lower():
            print(f"  [warn] per-channel requires opset>=13, falling back to per-tensor")
            reader2 = PalmVeinCalibrationReader(calib_images, input_name, args.input_size)
            quantize_static(
                model_input=str(fp32_path),
                model_output=str(int8_path),
                calibration_data_reader=reader2,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=False,
            )
        else:
            raise

    fp32_size = fp32_path.stat().st_size / 1e6
    int8_size = int8_path.stat().st_size / 1e6

    print(f"  FP32 size      : {fp32_size:.3f} MB")
    print(f"  INT8 size      : {int8_size:.3f} MB")

    print_section("BENCHMARK FP32 vs INT8")
    fp32_stats = benchmark_onnx(fp32_path, args.input_size, args.threads, args.warmup, args.runs)
    int8_stats = benchmark_onnx(int8_path, args.input_size, args.threads, args.warmup, args.runs)

    speedup = fp32_stats["mean_ms"] / int8_stats["mean_ms"] if int8_stats["mean_ms"] > 0 else float("nan")

    print(
        f"  FP32 ONNX 4T : {fp32_stats['mean_ms']:.2f} ms "
        f"(median={fp32_stats['median_ms']:.2f}, p95={fp32_stats['p95_ms']:.2f})"
    )
    print(
        f"  INT8 ONNX 4T : {int8_stats['mean_ms']:.2f} ms "
        f"(median={int8_stats['median_ms']:.2f}, p95={int8_stats['p95_ms']:.2f})"
    )
    print(f"  Speedup       : {speedup:.2f}x")

    results = {
        "model_dir": str(model_dir),
        "fp32_onnx": str(fp32_path),
        "int8_onnx": str(int8_path),
        "calib_dir": str(calib_dir),
        "num_calib": len(calib_images),
        "threads": args.threads,
        "fp32_size_mb": round(fp32_size, 4),
        "int8_size_mb": round(int8_size, 4),
        "fp32_4t_ms": round(fp32_stats["mean_ms"], 4),
        "int8_4t_ms": round(int8_stats["mean_ms"], 4),
        "speedup_x": round(speedup, 4),
    }

    out_path = model_dir / "benchmark_int8_static_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved  : {out_path}")


if __name__ == "__main__":
    main()

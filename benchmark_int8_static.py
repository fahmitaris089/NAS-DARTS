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


def ensure_min_opset(fp32_path: Path, min_opset: int = 13) -> Path:
    """Guarantee the ONNX uses opset >= min_opset so per-channel quant is valid.

    A model exported at opset < 13 silently disables per-channel weight
    quantization in onnxruntime, which catastrophically degrades models with
    wide activation ranges (e.g. MobileNetV3 h-swish + SE). We upgrade in place
    via onnx.version_converter and write an `*_op{min_opset}.onnx` sibling so the
    original artifact is preserved.
    """
    import onnx

    model = onnx.load(str(fp32_path))
    current = max((op.version for op in model.opset_import if op.domain in ("", "ai.onnx")), default=0)
    if current >= min_opset:
        return fp32_path

    print(f"  [opset] {fp32_path.name} is opset {current} < {min_opset}; upgrading for per-channel quant")
    upgraded = onnx.version_converter.convert_version(model, min_opset)
    onnx.checker.check_model(upgraded)
    up_path = fp32_path.with_name(fp32_path.stem + f"_op{min_opset}.onnx")
    onnx.save(upgraded, str(up_path))
    print(f"  [opset] upgraded model written: {up_path.name}")
    return up_path


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
    parser.add_argument(
        "--onnx_name",
        default="model_benchmark.onnx",
        help="FP32 ONNX filename inside --model_dir (e.g. mobilenetv3_benchmark.onnx)",
    )
    args = parser.parse_args()

    model_dir = ROOT / args.model_dir
    fp32_path = model_dir / args.onnx_name
    int8_path = fp32_path.with_name(fp32_path.stem + "_int8_static.onnx")
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

    # Guarantee opset >= 13 BEFORE quantizing. Without this, a low-opset model
    # silently disables per-channel weight quantization, producing an unfair
    # (degraded) INT8 baseline. We upgrade rather than fall back to per-tensor.
    quant_input_path = ensure_min_opset(fp32_path, min_opset=13)

    # ORT-recommended pre-processing (symbolic shape inference + graph cleanup).
    # This is important for complex graphs (e.g. MobileNetV3 SE/h-swish): it
    # avoids degenerate bias scales and improves PTQ accuracy. Skips gracefully
    # if the helper is unavailable in the installed onnxruntime version.
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        pre_path = quant_input_path.with_name(quant_input_path.stem + "_pre.onnx")
        quant_pre_process(str(quant_input_path), str(pre_path), skip_symbolic_shape=False)
        quant_input_path = pre_path
        print(f"  [pre]   quant pre-process done: {pre_path.name}")
    except Exception as exc:  # noqa: BLE001 - non-fatal optimization step
        print(f"  [pre]   quant_pre_process skipped ({exc})")

    fp32_sess = make_session(fp32_path, args.threads)
    input_name = fp32_sess.get_inputs()[0].name
    reader = PalmVeinCalibrationReader(calib_images, input_name, args.input_size)

    # Strict per-channel. NEVER silently downgrade to per-tensor: a quality drop
    # must be visible, not hidden. If this raises, fix the export instead.
    quantize_static(
        model_input=str(quant_input_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    quant_recipe = {
        "per_channel": True,
        "quant_format": "QDQ",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "quant_input_onnx": str(quant_input_path),
        "quant_pre_process": quant_input_path.name.endswith("_pre.onnx"),
    }

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
        "quant_recipe": quant_recipe,
    }

    out_path = model_dir / "benchmark_int8_static_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved  : {out_path}")


if __name__ == "__main__":
    main()

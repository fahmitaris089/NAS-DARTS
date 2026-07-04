#!/usr/bin/env python3
"""
Export all Teacher/training_results models to FP32 ONNX and static INT8 ONNX.

For each subfolder containing config.json + best_model.pth:
  - model_benchmark.onnx
  - model_benchmark_metadata.json
  - model_benchmark_int8_static.onnx
  - benchmark_int8_static_results.json

Example:
  cd Teacher
  python3 export_all_teacher_onnx_int8.py \
    --results-dir training_results \
    --calib-dir /workspace/preprocessed_results \
    --num-calib 200
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"onnx/onnxruntime quantization modules required: {exc}") from exc

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from model_factory import create_model, get_input_size  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_module_prefix(state_dict: dict) -> dict:
    if all(str(k).startswith("module.") for k in state_dict.keys()):
        return {str(k)[7:]: v for k, v in state_dict.items()}
    return state_dict


def build_model(model_name: str, weights_path: Path, num_classes: int) -> torch.nn.Module:
    model, _ = create_model(model_name, num_classes)
    state = torch.load(weights_path, map_location="cpu")
    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def export_onnx(model: torch.nn.Module, output_path: Path, input_size: int, opset: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    signature = inspect.signature(torch.onnx.export)
    if "dynamo" in signature.parameters:
        export_kwargs["dynamo"] = False
    if "external_data" in signature.parameters:
        export_kwargs["external_data"] = False

    torch.onnx.export(model, dummy, str(tmp_path), **export_kwargs)
    onnx_model = onnx.load(str(tmp_path))
    onnx.checker.check_model(onnx_model)
    tmp_path.replace(output_path)


def preprocess_bmp(path: Path, input_size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    rgb = np.stack([arr, arr, arr], axis=0)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(rgb.astype(np.float32), axis=0)


def collect_calibration_images(calib_dir: Path, limit: int) -> list[Path]:
    images = sorted(calib_dir.rglob("*.bmp"))
    if not images:
        raise FileNotFoundError(f"No .bmp files found under calibration dir: {calib_dir}")
    return images[: min(limit, len(images))]


class PalmVeinCalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], input_name: str, input_size: int):
        self.input_name = input_name
        self._data = [preprocess_bmp(path, input_size) for path in image_paths]
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self._data):
            return None
        arr = self._data[self._idx]
        self._idx += 1
        return {self.input_name: arr}


def make_session(model_path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(int(threads), 1)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


def benchmark_onnx(model_path: Path, input_size: int, threads: int, warmup: int, runs: int) -> dict:
    sess = make_session(model_path, threads)
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    for _ in range(max(warmup, 0)):
        sess.run(None, {input_name: dummy})
    times = []
    for _ in range(max(runs, 1)):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
    }


def quantize_int8(
    fp32_path: Path,
    int8_path: Path,
    calib_images: list[Path],
    input_size: int,
    threads: int,
    activation_type: QuantType,
    weight_type: QuantType,
    skip_first_conv: int,
) -> dict:
    sess = make_session(fp32_path, threads)
    input_name = sess.get_inputs()[0].name
    reader = PalmVeinCalibrationReader(calib_images, input_name, input_size)

    quant_input_path = fp32_path
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        pre_path = fp32_path.with_name(fp32_path.stem + "_pre.onnx")
        quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=False)
        quant_input_path = pre_path
    except Exception as exc:  # noqa: BLE001
        print(f"    [warn] quant_pre_process skipped: {exc}")

    nodes_to_quantize = None
    if skip_first_conv > 0:
        model = onnx.load(str(quant_input_path))
        conv_nodes = [node.name for node in model.graph.node if node.op_type == "Conv"]
        gemm_nodes = [node.name for node in model.graph.node if node.op_type == "Gemm"]
        nodes_to_quantize = conv_nodes[skip_first_conv:] + gemm_nodes
        print(
            f"    partial INT8: skipping first {skip_first_conv} Conv nodes, "
            f"quantizing {len(nodes_to_quantize)} Conv/Gemm nodes"
        )

    quantize_static(
        model_input=str(quant_input_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=activation_type,
        weight_type=weight_type,
        per_channel=True,
        nodes_to_quantize=nodes_to_quantize,
    )
    return {
        "per_channel": True,
        "quant_format": "QDQ",
        "activation_type": "QUInt8" if activation_type == QuantType.QUInt8 else "QInt8",
        "weight_type": "QUInt8" if weight_type == QuantType.QUInt8 else "QInt8",
        "quant_input_onnx": str(quant_input_path),
        "quant_pre_process": quant_input_path.name.endswith("_pre.onnx"),
        "skip_first_conv": skip_first_conv,
        "nodes_to_quantize": len(nodes_to_quantize) if nodes_to_quantize is not None else None,
    }


def infer_num_classes(model_dir: Path, config: dict) -> int:
    test_path = model_dir / "test_results.json"
    if test_path.exists():
        # Teacher experiment is fixed at 834 classes; keep this explicit fallback.
        return int(config.get("num_classes", 834))
    return int(config.get("num_classes", 834))


def process_model(model_dir: Path, args: argparse.Namespace, calib_images: list[Path]) -> bool:
    config_path = model_dir / "config.json"
    weights_path = model_dir / "best_model.pth"
    if not config_path.exists() or not weights_path.exists():
        print(f"[skip] {model_dir.name}: config.json/best_model.pth not found")
        return False

    config = load_json(config_path)
    model_name = str(config.get("model", model_dir.name))
    num_classes = infer_num_classes(model_dir, config)
    input_size = int(args.input_size or config.get("input_size", get_input_size(model_name)))
    fp32_path = model_dir / args.onnx_name
    int8_path = model_dir / args.int8_name
    meta_path = model_dir / "model_benchmark_metadata.json"
    result_path = model_dir / "benchmark_int8_static_results.json"

    print(f"\n=== {model_dir.name} ({model_name}) ===")
    print(f"  weights : {weights_path}")
    print(f"  input   : {input_size}x{input_size}, classes={num_classes}")

    if args.skip_existing and fp32_path.exists():
        print(f"  [skip] FP32 ONNX exists: {fp32_path.name}")
    else:
        model = build_model(model_name, weights_path, num_classes)
        export_onnx(model, fp32_path, input_size, args.opset)
        print(f"  FP32 ONNX: {fp32_path.name} ({fp32_path.stat().st_size / 1e6:.3f} MB)")

    metadata = {
        "exported_at": datetime.now().isoformat(),
        "model": model_name,
        "model_dir": str(model_dir),
        "weights_path": str(weights_path),
        "onnx_path": str(fp32_path),
        "num_classes": num_classes,
        "input_size": input_size,
        "opset": args.opset,
        "backend": "onnxruntime",
        "output_names": ["logits"],
        "logits_output_name": "logits",
        "model_size_mb": round(fp32_path.stat().st_size / 1e6, 6),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.skip_existing and int8_path.exists():
        print(f"  [skip] INT8 ONNX exists: {int8_path.name}")
        quant_recipe = None
    else:
        activation_type = QuantType.QUInt8 if args.activation_type == "QUInt8" else QuantType.QInt8
        weight_type = QuantType.QUInt8 if args.weight_type == "QUInt8" else QuantType.QInt8
        quant_recipe = quantize_int8(
            fp32_path,
            int8_path,
            calib_images,
            input_size,
            args.threads,
            activation_type,
            weight_type,
            args.skip_first_conv,
        )
        print(f"  INT8 ONNX: {int8_path.name} ({int8_path.stat().st_size / 1e6:.3f} MB)")

    fp32_stats = benchmark_onnx(fp32_path, input_size, args.threads, args.warmup, args.runs)
    int8_stats = benchmark_onnx(int8_path, input_size, args.threads, args.warmup, args.runs)
    speedup = fp32_stats["mean_ms"] / int8_stats["mean_ms"] if int8_stats["mean_ms"] > 0 else float("nan")

    results = {
        "model_dir": str(model_dir),
        "model": model_name,
        "fp32_onnx": str(fp32_path),
        "int8_onnx": str(int8_path),
        "calib_dir": str(args.calib_dir),
        "num_calib": len(calib_images),
        "threads": args.threads,
        "fp32_size_mb": round(fp32_path.stat().st_size / 1e6, 4),
        "int8_size_mb": round(int8_path.stat().st_size / 1e6, 4),
        "fp32_4t_ms": round(fp32_stats["mean_ms"], 4),
        "int8_4t_ms": round(int8_stats["mean_ms"], 4),
        "speedup_x": round(speedup, 4),
        "quant_recipe": quant_recipe,
    }
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  speedup : {speedup:.2f}x")
    print(f"  results : {result_path.name}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all teacher models to ONNX + INT8")
    parser.add_argument("--results-dir", type=Path, default=HERE / "training_results")
    parser.add_argument("--calib-dir", type=Path, default=PROJECT_ROOT / "preprocessed_results")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Optional subset of folder names/model names to export")
    parser.add_argument("--num-calib", type=int, default=200)
    parser.add_argument("--input-size", type=int, default=None,
                        help="Override input size for all models. Default: from model factory/config.")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--onnx-name", default="model_benchmark.onnx")
    parser.add_argument("--int8-name", default="model_benchmark_int8_static.onnx")
    parser.add_argument(
        "--activation-type",
        choices=["QUInt8", "QInt8"],
        default="QUInt8",
        help="Static PTQ activation type. QUInt8 is safer for torchvision/timm CPU baselines.",
    )
    parser.add_argument(
        "--weight-type",
        choices=["QInt8", "QUInt8"],
        default="QInt8",
        help="Static PTQ weight type.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--skip-first-conv",
        type=int,
        default=0,
        help=(
            "Partial INT8 recipe: leave the first N Conv nodes in FP32 and quantize "
            "remaining Conv/Gemm nodes. Useful for quantization-sensitive MobileNetV3Small."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir = args.results_dir.resolve()
    args.calib_dir = args.calib_dir.resolve()
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {args.results_dir}")
    if not args.calib_dir.exists():
        raise FileNotFoundError(f"Calibration dir not found: {args.calib_dir}")

    calib_images = collect_calibration_images(args.calib_dir, args.num_calib)
    print(f"Results dir : {args.results_dir}")
    print(f"Calib dir   : {args.calib_dir}")
    print(f"Calib imgs  : {len(calib_images)}")

    selected = set(args.models or [])
    model_dirs = sorted([p for p in args.results_dir.iterdir() if p.is_dir()])
    done = 0
    for model_dir in model_dirs:
        if selected and model_dir.name not in selected:
            cfg_path = model_dir / "config.json"
            model_name = load_json(cfg_path).get("model") if cfg_path.exists() else None
            if model_name not in selected:
                continue
        try:
            done += int(process_model(model_dir, args, calib_images))
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {model_dir.name}: {exc}")
            if not selected:
                continue
            raise

    print(f"\nDone. Exported/processed {done} model(s).")


if __name__ == "__main__":
    main()

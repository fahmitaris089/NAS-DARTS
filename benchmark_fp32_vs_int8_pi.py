"""
Benchmark FP32 vs INT8 untuk SATU model (akurasi + latency + size) di Pi.

Menunjuk satu folder model yang berisi:
  - model_benchmark.onnx            (FP32)
  - model_benchmark_int8_static.onnx (INT8 static, per-channel)
lalu mengukur keduanya pada test split yang sama dan melaporkan delta.

Hanya butuh onnxruntime (tanpa torch) → aman dijalankan DI PI.

Contoh:
  python3 benchmark_fp32_vs_int8_pi.py \
      --model-dir nas_results/retrain_hwNAS_l0.20_C8_stemds4 \
      --data-dir preprocessed_results \
      --split-path split_info.json \
      --threads 4 --max-samples 834
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
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"onnxruntime diperlukan: {exc}") from exc

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FP32 vs INT8 benchmark untuk satu model")
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Folder berisi model_benchmark.onnx & model_benchmark_int8_static.onnx")
    p.add_argument("--fp32-name", default="model_benchmark.onnx")
    p.add_argument("--int8-name", default="model_benchmark_int8_static.onnx")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split-path", type=Path, required=True)
    p.add_argument("--label", type=str, default=None, help="Label model (default: nama folder)")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=0, help="0 = pakai seluruh test split")
    p.add_argument("--save-path", type=Path, default=None)
    return p.parse_args()


def load_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Split tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_label_map(subjects: list[str]) -> dict[str, int]:
    ordered = sorted((str(s) for s in subjects), key=int)
    return {s: i for i, s in enumerate(ordered)}


def build_test_samples(data_dir: Path, split: dict[str, Any], max_samples: int):
    samples, missing = [], 0
    for subj, fname in split["test"]:
        flat = data_dir / str(subj) / fname
        if flat.exists():
            samples.append((flat, str(subj)))
        else:
            sd = data_dir / str(subj)
            found = list(sd.rglob(fname)) if sd.exists() else []
            if found:
                samples.append((found[0], str(subj)))
            else:
                missing += 1
    if missing:
        print(f"  [warn] {missing} file test tak ditemukan (dilewati)")
    if not samples:
        raise FileNotFoundError(f"Tidak ada sampel test di {data_dir}")
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def preprocess(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    g = np.asarray(img, dtype=np.float32) / 255.0
    rgb = np.stack([g, g, g], axis=0)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(rgb.astype(np.float32), axis=0)


def make_session(model_path: Path, threads: int) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX tak ditemukan: {model_path}")
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(int(threads), 1)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


def run_model(model_path: Path, samples, cached, label_map, threads, warmup) -> dict[str, Any]:
    sess = make_session(model_path, threads)
    iname = sess.get_inputs()[0].name
    dummy = cached[0]
    for _ in range(max(warmup, 0)):
        sess.run(None, {iname: dummy})

    lat, correct = [], 0
    for (path, subj), arr in zip(samples, cached):
        t0 = time.perf_counter()
        out = sess.run(None, {iname: arr})
        lat.append((time.perf_counter() - t0) * 1000.0)
        if subj in label_map and int(np.argmax(out[0][0])) == label_map[subj]:
            correct += 1
    a = np.asarray(lat, dtype=np.float64)
    return {
        "onnx": str(model_path),
        "accuracy": correct / len(samples),
        "correct": correct,
        "n": len(samples),
        "size_mb": model_path.stat().st_size / 1e6,
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
    }


def show(tag: str, r: dict[str, Any]) -> None:
    print(f"\n[{tag}]  {Path(r['onnx']).name}")
    print(f"  accuracy : {r['accuracy']*100:.2f}% ({r['correct']}/{r['n']})")
    print(f"  size     : {r['size_mb']:.3f} MB")
    print(f"  latency  : mean={r['mean_ms']:.2f} ms  median={r['median_ms']:.2f} ms  p95={r['p95_ms']:.2f} ms")


def main() -> None:
    args = parse_args()
    label = args.label or args.model_dir.name
    split = load_split(args.split_path)
    label_map = build_label_map(split["subjects"])
    samples = build_test_samples(args.data_dir, split, args.max_samples)
    cached = [preprocess(p, args.input_size) for p, _ in samples]

    print(f"Model    : {label}")
    print(f"Test imgs: {len(samples)} | threads: {args.threads}")

    fp32 = run_model(args.model_dir / args.fp32_name, samples, cached, label_map, args.threads, args.warmup)
    int8 = run_model(args.model_dir / args.int8_name, samples, cached, label_map, args.threads, args.warmup)
    show("FP32", fp32)
    show("INT8", int8)

    speedup = fp32["mean_ms"] / int8["mean_ms"] if int8["mean_ms"] > 0 else float("nan")
    d_acc = (int8["accuracy"] - fp32["accuracy"]) * 100.0
    print("\n── Delta (INT8 vs FP32) ──")
    print(f"  latency : {speedup:.2f}x  ({'INT8 lebih cepat' if speedup>1 else 'INT8 lebih LAMBAT'})")
    print(f"  size    : {fp32['size_mb']/int8['size_mb']:.2f}x lebih kecil")
    print(f"  akurasi : {d_acc:+.2f} pp")

    out = {
        "label": label, "model_dir": str(args.model_dir),
        "num_samples": len(samples), "threads": args.threads,
        "fp32": fp32, "int8": int8,
        "int8_vs_fp32": {
            "latency_speedup_x": speedup,
            "size_ratio_x": fp32["size_mb"] / int8["size_mb"],
            "accuracy_delta_pp": d_acc,
            "int8_faster": speedup > 1.0,
        },
    }
    save = args.save_path or (args.model_dir / "benchmark_fp32_vs_int8_pi.json")
    Path(save).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults saved: {save}")


if __name__ == "__main__":
    main()

"""
Benchmark MobileNetV3Large — Palm Vein Recognition
====================================================
Mengukur latency, FLOPs, dan memproyeksikan ke Raspberry Pi 4/5.

Usage:
    cd /Users/fahmitaris/Downloads/NAS-DARTS
    python3 benchmark_mobilenet.py
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

_ROOT = Path(__file__).resolve().parent

# ── Scaling factors Mac ARM → RPi (sama dengan benchmark_rpi.py) ─────────────
SCALING = {
    "rpi4": {"pytorch_1t": 18.0, "pytorch_4t": 10.0, "onnx_1t": 10.0, "onnx_4t": 6.0},
    "rpi5": {"pytorch_1t":  6.5, "pytorch_4t":  4.0, "onnx_1t":  3.5, "onnx_4t": 2.2},
}


def load_mobilenet(weights_path: Path, num_classes: int = 834) -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    # Ganti classifier head sesuai num_classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    state_dict = torch.load(weights_path, map_location="cpu")
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def compute_flops(model, input_size=224):
    try:
        from thop import profile, clever_format
        dummy = torch.randn(1, 3, input_size, input_size)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        macs_s, params_s = clever_format([macs, params], "%.3f")
        return macs, macs_s, params_s
    except Exception as e:
        return None, f"Error: {e}", "N/A"


def measure_pytorch(model, input_size, threads, warmup=20, runs=100):
    torch.set_num_threads(threads)
    dummy = torch.randn(1, 3, input_size, input_size)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {"mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "std_ms": float(np.std(times))}


def export_onnx(model, input_size, out_path):
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(model, dummy, str(out_path),
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=13, do_constant_folding=True)
    return out_path.stat().st_size / 1e6


def measure_onnx(onnx_path, input_size, threads, warmup=20, runs=100):
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {inp_name: dummy})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {"mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "std_ms": float(np.std(times))}


def project(stats, rpi_key, scale_key):
    if stats is None:
        return None
    f = SCALING[rpi_key][scale_key]
    return {"mean_ms": stats["mean_ms"] * f, "p95_ms": stats["p95_ms"] * f, "factor": f}


def fmt(stats, label, width=32):
    if stats is None:
        return f"  {label:{width}}: N/A"
    return (f"  {label:{width}}: {stats['mean_ms']:7.1f} ms  "
            f"(median={stats['median_ms']:.1f}, p95={stats['p95_ms']:.1f})")


def section(title):
    print(f"\n{'═'*60}\n  {title}\n{'═'*60}")


def main():
    MODEL_DIR  = _ROOT / "MobileNetV3Large"
    WEIGHTS    = MODEL_DIR / "best_model.pth"
    ONNX_PATH  = MODEL_DIR / "mobilenetv3_benchmark.onnx"
    INPUT_SIZE = 224
    THREADS    = 4
    WARMUP     = 20
    RUNS       = 100

    is_arm_mac = platform.machine().lower() == "arm64" and sys.platform == "darwin"
    host_label = "Mac ARM (Apple Silicon)" if is_arm_mac else platform.processor()

    with open(MODEL_DIR / "config.json") as f:
        cfg = json.load(f)
    with open(MODEL_DIR / "test_results.json") as f:
        results = json.load(f)

    section("KONFIGURASI")
    print(f"  Model          : MobileNetV3Large")
    print(f"  Weights        : {WEIGHTS.name}")
    print(f"  Host           : {host_label}")
    print(f"  Input size     : {INPUT_SIZE}×{INPUT_SIZE}")
    print(f"  Threads        : {THREADS}")
    print(f"  Warmup/Runs    : {WARMUP}/{RUNS}")

    section("PERFORMA MODEL (dari test_results.json)")
    print(f"  Test Accuracy  : {results['accuracy']*100:.2f}%")
    print(f"  EER            : {results['eer']*100:.4f}%")
    print(f"  AUC            : {results['auc']:.6f}")
    print(f"  Best Epoch     : {results['best_epoch']}")

    # Load
    section("LOAD MODEL")
    print("  Memuat MobileNetV3Large... ", end="", flush=True)
    model = load_mobilenet(WEIGHTS, num_classes=cfg["total_params"] and 834)
    print("OK")

    total_p = count_params(model)
    pth_mb  = WEIGHTS.stat().st_size / 1e6
    print(f"\n  Total params   : {total_p:,}  ({total_p/1e6:.3f} M)")
    print(f"  .pth size      : {pth_mb:.2f} MB")

    # FLOPs
    section("FLOPs / MACs")
    macs, macs_s, params_s = compute_flops(model, INPUT_SIZE)
    print(f"  MACs (FLOPs/2) : {macs_s}")
    if macs:
        print(f"  Total FLOPs    : {macs*2/1e9:.2f} GFLOPs")
        print(f"  Teoritis RPi4 @ 1 GFLOPS  : {macs*2/1e9*1000:.0f} ms")
        print(f"  Teoritis RPi5 @ 3.5 GFLOPS: {macs*2/1e9/3.5*1000:.0f} ms")

    # PyTorch benchmark
    section(f"PYTORCH CPU BENCHMARK ({host_label})")
    print(f"\n  [1/2] 1-thread ({RUNS} runs)...")
    st_1t = measure_pytorch(model, INPUT_SIZE, 1, WARMUP, RUNS)
    print(fmt(st_1t, "PyTorch 1T"))

    print(f"\n  [2/2] {THREADS}-threads ({RUNS} runs)...")
    st_4t = measure_pytorch(model, INPUT_SIZE, THREADS, WARMUP, RUNS)
    print(fmt(st_4t, f"PyTorch {THREADS}T"))

    # ONNX
    section("ONNX EXPORT & BENCHMARK")
    print("  Exporting... ", end="", flush=True)
    try:
        onnx_mb = export_onnx(model, INPUT_SIZE, ONNX_PATH)
        print(f"OK  ({onnx_mb:.2f} MB)")

        print(f"\n  [1/2] ONNX 1-thread ({RUNS} runs)...")
        so_1t = measure_onnx(ONNX_PATH, INPUT_SIZE, 1, WARMUP, RUNS)
        print(fmt(so_1t, "ONNX Runtime 1T"))

        print(f"\n  [2/2] ONNX {THREADS}-threads ({RUNS} runs)...")
        so_4t = measure_onnx(ONNX_PATH, INPUT_SIZE, THREADS, WARMUP, RUNS)
        print(fmt(so_4t, f"ONNX Runtime {THREADS}T"))
    except Exception as e:
        print(f"GAGAL: {e}")
        so_1t = so_4t = None

    # Proyeksi RPi
    section("ESTIMASI RASPBERRY PI  ⚠️  (projected)")
    for rpi_key in ["rpi4", "rpi5"]:
        info = {"rpi4": "RPi 4B (A72 @ 1.5GHz) [4GB]",
                "rpi5": "RPi 5  (A76 @ 2.4GHz) [4GB]"}[rpi_key]
        print(f"\n  ┌─ {info}")
        for label, stats, skey in [
            ("PyTorch 4T",      st_4t,  "pytorch_4t"),
            ("ONNX Runtime 1T", so_1t,  "onnx_1t"),
            ("ONNX Runtime 4T", so_4t,  "onnx_4t"),
        ]:
            pr = project(stats, rpi_key, skey)
            if pr:
                fps = 1000 / pr["mean_ms"]
                print(f"  │  {label:20s}: ~{pr['mean_ms']:5.0f} ms  (~{fps:.2f} FPS)")
        print(f"  └{'─'*54}")

    # Comparison vs P-DARTS KD
    section("PERBANDINGAN vs P-DARTS KD STUDENT")
    # P-DARTS best KD values (from previous benchmark)
    pdarts_onnx4t_ms = 11.0   # dari benchmark_rpi.py run sebelumnya

    mv3_onnx4t_ms_rpi5 = so_4t["mean_ms"] * SCALING["rpi5"]["onnx_4t"] if so_4t else None
    mv3_onnx4t_ms_rpi4 = so_4t["mean_ms"] * SCALING["rpi4"]["onnx_4t"] if so_4t else None
    pdarts_rpi5_ms = pdarts_onnx4t_ms * SCALING["rpi5"]["onnx_4t"]
    pdarts_rpi4_ms = pdarts_onnx4t_ms * SCALING["rpi4"]["onnx_4t"]

    print(f"\n  {'Model':<35} {'Params':>10}  {'Test Acc':>9}  {'EER':>8}  {'RPi5 ONNX4T':>13}  {'RPi4 ONNX4T':>13}")
    print(f"  {'─'*100}")
    print(f"  {'P-DARTS KD (best, e500)':<35} {'82,606':>10}  {'98.92%':>9}  {'0.022%':>8}  {'~'+str(round(pdarts_rpi5_ms))+'ms':>13}  {'~'+str(round(pdarts_rpi4_ms))+'ms':>13}")
    if mv3_onnx4t_ms_rpi5:
        print(f"  {'MobileNetV3Large':<35} {total_p:>10,}  {results['accuracy']*100:>8.2f}%  {results['eer']*100:>7.4f}%  {'~'+str(round(mv3_onnx4t_ms_rpi5))+'ms':>13}  {'~'+str(round(mv3_onnx4t_ms_rpi4))+'ms':>13}")

    if mv3_onnx4t_ms_rpi5 and mv3_onnx4t_ms_rpi4:
        slower_rpi5 = mv3_onnx4t_ms_rpi5 / pdarts_rpi5_ms
        slower_rpi4 = mv3_onnx4t_ms_rpi4 / pdarts_rpi4_ms
        bigger = total_p / 82606
        print(f"\n  MobileNetV3Large vs P-DARTS KD:")
        print(f"    Params  : {bigger:.1f}× lebih besar ({total_p:,} vs 82,606)")
        print(f"    RPi 5   : {slower_rpi5:.1f}× lebih lambat")
        print(f"    RPi 4   : {slower_rpi4:.1f}× lebih lambat")
        print(f"    Test Acc: MobileNet {results['accuracy']*100:.2f}% vs P-DARTS KD 98.92%")
        print(f"    EER     : MobileNet {results['eer']*100:.4f}% vs P-DARTS KD 0.022%")

    # Save summary
    summary = {
        "model": "MobileNetV3Large",
        "total_params": total_p,
        "pth_size_mb": round(pth_mb, 3),
        "test_acc": results["accuracy"],
        "eer_pct": results["eer"] * 100,
        "auc": results["auc"],
        "flops_gflops": round(macs * 2 / 1e9, 3) if macs else None,
        "host": host_label,
        "pytorch_1t_ms": round(st_1t["mean_ms"], 2),
        f"pytorch_{THREADS}t_ms": round(st_4t["mean_ms"], 2),
        "onnx_1t_ms": round(so_1t["mean_ms"], 2) if so_1t else None,
        f"onnx_{THREADS}t_ms": round(so_4t["mean_ms"], 2) if so_4t else None,
        "estimates": {
            "rpi4": {
                "onnx_4t_ms": round(so_4t["mean_ms"] * SCALING["rpi4"]["onnx_4t"]) if so_4t else None
            },
            "rpi5": {
                "onnx_4t_ms": round(so_4t["mean_ms"] * SCALING["rpi5"]["onnx_4t"]) if so_4t else None
            },
        }
    }
    out = MODEL_DIR / "benchmark_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Hasil disimpan ke: {out}")


if __name__ == "__main__":
    main()

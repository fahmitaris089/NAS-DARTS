"""
Raspberry Pi Inference Benchmark — P-DARTS KD Student Model
=============================================================
Mengukur latency, parameter count, MACs/FLOPs, dan memory footprint model,
lalu memproyeksikan estimasi kecepatan di Raspberry Pi 4 dan Pi 5.

Usage:
    cd /Users/fahmitaris/Downloads/NAS-DARTS
    python3 benchmark_rpi.py

    # Custom model path:
    python3 benchmark_rpi.py --model_dir knowledge_distilation/kd_results/run5_efficientNetV2M_t10_a0.5_e500

Options:
    --model_dir     Path ke folder hasil KD (berisi best_model.pth + config.json)
    --export_onnx   Export ke ONNX dan benchmark ONNX Runtime juga
    --warmup        Jumlah warmup runs (default: 20)
    --runs          Jumlah timed runs untuk latency (default: 100)
    --threads       Jumlah CPU threads (default: 4, sesuai RPi)
    --input_size    Ukuran input gambar (default: 224)
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Pastikan project root ada di path ─────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from genotypes import dict_to_genotype
from model_eval import EvalNetwork

# ─────────────────────────────────────────────────────────────────────────────
# Scaling factors: Mac ARM (M-series) → RPi
# Sumber: Geekbench 5 single-core ratios + ML inference benchmarks komunitas
# Mac M1/M2 single-core ~2400-2500, RPi4 ~230, RPi5 ~620 (Geekbench 5)
# Untuk ML workload PyTorch CPU, faktor empiris lebih konservatif
# ─────────────────────────────────────────────────────────────────────────────
SCALING = {
    # (slowdown_factor_1T, slowdown_factor_4T)
    # slowdown: berapa kali lebih lambat RPi vs Mac ARM (per-core perf ratio)
    "rpi4": {
        "name": "Raspberry Pi 4B (Cortex-A72 @ 1.5GHz)",
        "ram": "4 GB LPDDR4",
        "pytorch_1t":  18.0,   # ~18x lebih lambat dari M-series 1 thread
        "pytorch_4t":  10.0,   # thermal throttle + memory BW limit → tidak linear
        "onnx_1t":     10.0,   # ONNX Runtime lebih efisien pada ARM
        "onnx_4t":      6.0,
    },
    "rpi5": {
        "name": "Raspberry Pi 5 (Cortex-A76 @ 2.4GHz)",
        "ram": "4 GB LPDDR4X",
        "pytorch_1t":   6.5,   # A76 jauh lebih modern, ~2.7x/core vs A72
        "pytorch_4t":   4.0,
        "onnx_1t":      3.5,
        "onnx_4t":      2.2,
    },
}

# Untuk mesin bukan Mac ARM (Intel Mac atau Linux x86), scaling berbeda
SCALING_X86 = {
    "rpi4": {"pytorch_1t": 8.0,  "pytorch_4t": 5.0,  "onnx_1t": 4.5,  "onnx_4t": 3.0},
    "rpi5": {"pytorch_1t": 3.0,  "pytorch_4t": 2.0,  "onnx_1t": 1.8,  "onnx_4t": 1.2},
}


# ─────────────────────────────────────────────────────────────────────────────

def detect_host():
    """Deteksi mesin host untuk memilih scaling factor yang tepat."""
    machine = platform.machine().lower()
    processor = platform.processor().lower()
    is_arm_mac = (machine == "arm64" and sys.platform == "darwin")
    is_intel_mac = ("x86" in machine or "intel" in processor) and sys.platform == "darwin"
    return is_arm_mac, is_intel_mac


def load_model(model_dir: Path, device: torch.device) -> nn.Module:
    """Load student model dari folder KD result."""
    config_path = model_dir / "config.json"
    weights_path = model_dir / "best_model.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json tidak ditemukan di {model_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"best_model.pth tidak ditemukan di {model_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    # Load genotype — bisa dari cfg sendiri (retrain) atau dari student_config_path (KD)
    if "genotype" in cfg:
        # Config ini sudah berisi genotype langsung (retrain result)
        student_cfg = cfg
    else:
        student_cfg_path = cfg.get("student_config_path", "")
        student_cfg_full = _ROOT / student_cfg_path if student_cfg_path else None
        if not student_cfg_full or not student_cfg_full.is_file():
            # Fallback: coba path alternatif
            for candidate in [
                _ROOT / "nas_results/retrain_run5/config.json",
                _ROOT / "nas_results/retrain/config.json",
            ]:
                if candidate.exists():
                    student_cfg_full = candidate
                    break
        with open(student_cfg_full) as f:
            student_cfg = json.load(f)

    genotype = dict_to_genotype(student_cfg["genotype"])
    C_init = cfg.get("student_C_init", student_cfg.get("C_init", 4))
    num_cells = cfg.get("student_num_cells", student_cfg.get("num_cells", 8))
    num_classes = cfg.get("num_classes", 834)
    dropout = cfg.get("student_dropout", 0.3)

    model = EvalNetwork(
        genotype=genotype,
        C_init=C_init,
        num_cells=num_cells,
        num_classes=num_classes,
        auxiliary=False,   # auxiliary head tidak dipakai saat inference
        dropout=dropout,
    )

    state_dict = torch.load(weights_path, map_location="cpu")
    # Handle key prefix mismatch
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model, C_init, num_cells, num_classes


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_flops(model: nn.Module, input_size: int):
    """Hitung MACs dan params menggunakan thop."""
    try:
        from thop import profile, clever_format
        dummy = torch.randn(1, 3, input_size, input_size)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        macs_str, params_str = clever_format([macs, params], "%.3f")
        return macs, params, macs_str, params_str
    except ImportError:
        return None, None, "N/A (install thop)", "N/A"
    except Exception as e:
        return None, None, f"Error: {e}", "N/A"


def measure_latency_pytorch(model: nn.Module, input_size: int,
                             num_threads: int, warmup: int, runs: int):
    """Ukur latency PyTorch CPU inference."""
    torch.set_num_threads(num_threads)
    dummy = torch.randn(1, 3, input_size, input_size)

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(dummy)

        # Timed runs
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    return {
        "mean_ms":   float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "std_ms":    float(np.std(times)),
        "p95_ms":    float(np.percentile(times, 95)),
        "min_ms":    float(np.min(times)),
        "max_ms":    float(np.max(times)),
    }


def export_onnx(model: nn.Module, input_size: int, output_path: Path):
    """Export model ke ONNX format."""
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=13,
        do_constant_folding=True,
    )
    size_mb = output_path.stat().st_size / 1e6
    return size_mb


def measure_latency_onnx(onnx_path: Path, input_size: int,
                          num_threads: int, warmup: int, runs: int):
    """Ukur latency ONNX Runtime inference."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None, "onnxruntime not installed"

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = num_threads
    sess_opts.inter_op_num_threads = 1
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(str(onnx_path), sess_opts,
                                providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms":   float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "std_ms":    float(np.std(times)),
        "p95_ms":    float(np.percentile(times, 95)),
        "min_ms":    float(np.min(times)),
    }, None


def model_size_mb(weights_path: Path):
    return weights_path.stat().st_size / 1e6


def print_section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def print_latency(label, stats):
    if stats is None:
        print(f"  {label:30s}: N/A")
        return
    print(f"  {label:30s}: {stats['mean_ms']:7.1f} ms  "
          f"(median={stats['median_ms']:.1f}, p95={stats['p95_ms']:.1f}, "
          f"std={stats['std_ms']:.1f})")


def estimate_rpi(host_stats, scale_key, scale_factor_key, label):
    """Hitung estimasi RPi latency dari hasil host."""
    if host_stats is None:
        return None
    factor = SCALING[scale_key][scale_factor_key]
    mean  = host_stats["mean_ms"]   * factor
    p95   = host_stats["p95_ms"]    * factor
    return {"mean_ms": mean, "p95_ms": p95, "scale_factor": factor, "label": label}


def print_estimate(rpi_stat, source_label):
    if rpi_stat is None:
        return
    fps = 1000.0 / rpi_stat["mean_ms"] if rpi_stat["mean_ms"] > 0 else 0
    print(f"    ├─ mean : {rpi_stat['mean_ms']:7.0f} ms  (~{fps:.1f} FPS)")
    print(f"    └─ p95  : {rpi_stat['p95_ms']:7.0f} ms  "
          f"(scaling factor ×{rpi_stat['scale_factor']:.1f} dari {source_label})")


def main():
    parser = argparse.ArgumentParser(description="RPi Inference Benchmark")
    parser.add_argument("--model_dir", default=
        "knowledge_distilation/kd_results/run5_efficientNetV2M_t10_a0.5_e500",
        help="Path ke folder model (relatif dari project root)")
    parser.add_argument("--export_onnx", action="store_true", default=True,
        help="Export ke ONNX dan benchmark ONNX Runtime")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs",   type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=224)
    args = parser.parse_args()

    model_dir = _ROOT / args.model_dir
    if not model_dir.exists():
        sys.exit(f"[ERROR] Model directory tidak ditemukan: {model_dir}")

    is_arm_mac, is_intel_mac = detect_host()
    scaling_table = SCALING if (is_arm_mac or is_intel_mac) else SCALING_X86
    host_label = (
        "Mac ARM (Apple Silicon)"  if is_arm_mac else
        "Mac Intel"                 if is_intel_mac else
        platform.processor() or "unknown"
    )

    print_section("KONFIGURASI BENCHMARK")
    print(f"  Model dir    : {model_dir}")
    print(f"  Host machine : {host_label}")
    print(f"  PyTorch ver  : {torch.__version__}")
    print(f"  Input size   : {args.input_size}×{args.input_size}")
    print(f"  Threads      : {args.threads}")
    print(f"  Warmup/Runs  : {args.warmup}/{args.runs}")

    # ── Load Model ─────────────────────────────────────────────────────────
    print_section("LOAD MODEL")
    print("  Memuat model... ", end="", flush=True)
    device = torch.device("cpu")
    model, C_init, num_cells, num_classes = load_model(model_dir, device)
    model.eval()
    print("OK")

    # ── Parameter Count ─────────────────────────────────────────────────────
    total_params, train_params = count_params(model)
    weights_path = model_dir / "best_model.pth"
    pth_size = model_size_mb(weights_path)

    print(f"\n  Arsitektur   : P-DARTS EvalNetwork")
    print(f"  C_init       : {C_init}  |  num_cells: {num_cells}  |  classes: {num_classes}")
    print(f"  Total params : {total_params:,}  ({total_params/1e6:.4f} M)")
    print(f"  .pth size    : {pth_size:.2f} MB")

    # ── FLOPs / MACs ────────────────────────────────────────────────────────
    print_section("FLOPs / MACs ANALYSIS")
    macs, _, macs_str, params_str = compute_flops(model, args.input_size)
    print(f"  MACs (FLOPs/2): {macs_str}")
    print(f"  Total FLOPs   : {macs*2/1e6:.1f} MFLOPs" if macs else "  FLOPs: N/A")
    print(f"  Params (thop) : {params_str}")

    if macs:
        # Teoritis: berapa lama pada hardware tertentu
        # RPi4 effective throughput PyTorch ~0.5-1.5 GFLOPS (4T)
        # RPi5 effective throughput PyTorch ~2-5 GFLOPS (4T)
        rpi4_theoretical_ms = (macs * 2) / (1.0e9) * 1000   # assume 1 GFLOPS rpi4
        rpi5_theoretical_ms = (macs * 2) / (3.5e9) * 1000   # assume 3.5 GFLOPS rpi5
        print(f"\n  Teoritis @ 1 GFLOPS eff (RPi4 approx) : {rpi4_theoretical_ms:.1f} ms")
        print(f"  Teoritis @ 3.5 GFLOPS eff (RPi5 approx): {rpi5_theoretical_ms:.1f} ms")

    # ── PyTorch CPU Benchmark ───────────────────────────────────────────────
    print_section(f"PYTORCH CPU BENCHMARK ({host_label})")

    print(f"\n  [1/2] Mengukur dengan 1 thread ({args.runs} runs)...")
    stats_pt_1t = measure_latency_pytorch(model, args.input_size, 1,
                                           args.warmup, args.runs)
    print_latency("PyTorch 1-thread", stats_pt_1t)

    print(f"\n  [2/2] Mengukur dengan {args.threads} threads ({args.runs} runs)...")
    stats_pt_4t = measure_latency_pytorch(model, args.input_size, args.threads,
                                           args.warmup, args.runs)
    print_latency(f"PyTorch {args.threads}-threads", stats_pt_4t)

    # ── ONNX Export & Benchmark ─────────────────────────────────────────────
    stats_onnx_1t = None
    stats_onnx_4t = None
    onnx_path = model_dir / "model_benchmark.onnx"

    if args.export_onnx:
        print_section("ONNX EXPORT & BENCHMARK")
        print("  Exporting ke ONNX... ", end="", flush=True)
        try:
            onnx_size = export_onnx(model, args.input_size, onnx_path)
            print(f"OK  ({onnx_size:.2f} MB) → {onnx_path.name}")

            print(f"\n  [1/2] ONNX Runtime 1-thread ({args.runs} runs)...")
            stats_onnx_1t, err = measure_latency_onnx(onnx_path, args.input_size,
                                                        1, args.warmup, args.runs)
            if err:
                print(f"  Gagal: {err}")
            else:
                print_latency("ONNX Runtime 1-thread", stats_onnx_1t)

            print(f"\n  [2/2] ONNX Runtime {args.threads}-threads ({args.runs} runs)...")
            stats_onnx_4t, err = measure_latency_onnx(onnx_path, args.input_size,
                                                        args.threads, args.warmup,
                                                        args.runs)
            if err:
                print(f"  Gagal: {err}")
            else:
                print_latency(f"ONNX Runtime {args.threads}-threads", stats_onnx_4t)

        except Exception as e:
            print(f"GAGAL: {e}")

    # ── Proyeksi RPi ────────────────────────────────────────────────────────
    print_section("ESTIMASI RASPBERRY PI  ⚠️  (projected, bukan diukur langsung)")
    print("  Catatan: Estimasi berdasarkan scaling factor empiris.")
    print("  Gunakan sebagai patokan kasar, bukan angka pasti.\n")
    sc = scaling_table

    for rpi_key in ["rpi4", "rpi5"]:
        rpi_info = SCALING[rpi_key]
        print(f"  ┌─ {rpi_info['name']}  [{rpi_info['ram']}]")

        # PyTorch 1T
        est = estimate_rpi(stats_pt_1t, rpi_key, "pytorch_1t", host_label)
        if est:
            fps = 1000.0 / est["mean_ms"]
            print(f"  │  PyTorch CPU 1T  : ~{est['mean_ms']:.0f} ms  (~{fps:.2f} FPS)")

        # PyTorch 4T
        est4 = estimate_rpi(stats_pt_4t, rpi_key, "pytorch_4t", host_label)
        if est4:
            fps4 = 1000.0 / est4["mean_ms"]
            print(f"  │  PyTorch CPU 4T  : ~{est4['mean_ms']:.0f} ms  (~{fps4:.2f} FPS)")

        # ONNX 1T
        est_o1 = estimate_rpi(stats_onnx_1t, rpi_key, "onnx_1t", host_label)
        if est_o1:
            fps_o1 = 1000.0 / est_o1["mean_ms"]
            print(f"  │  ONNX Runtime 1T : ~{est_o1['mean_ms']:.0f} ms  (~{fps_o1:.2f} FPS)")

        # ONNX 4T
        est_o4 = estimate_rpi(stats_onnx_4t, rpi_key, "onnx_4t", host_label)
        if est_o4:
            fps_o4 = 1000.0 / est_o4["mean_ms"]
            print(f"  │  ONNX Runtime 4T : ~{est_o4['mean_ms']:.0f} ms  (~{fps_o4:.2f} FPS)")

        print(f"  └{'─'*56}")

    # ── Rekomendasi ─────────────────────────────────────────────────────────
    print_section("REKOMENDASI MINIMUM SPEC")

    # Hitung estimasi terbaik pakai ONNX 4T
    best_rpi4 = estimate_rpi(stats_onnx_4t or stats_pt_4t,
                              "rpi4", "onnx_4t" if stats_onnx_4t else "pytorch_4t",
                              host_label)
    best_rpi5 = estimate_rpi(stats_onnx_4t or stats_pt_4t,
                              "rpi5", "onnx_4t" if stats_onnx_4t else "pytorch_4t",
                              host_label)

    if best_rpi4:
        rpi4_ms = best_rpi4["mean_ms"]
        rpi5_ms = best_rpi5["mean_ms"]

        print(f"\n  Model: P-DARTS Student (KD) — {total_params:,} params")
        print(f"  Input: {args.input_size}×{args.input_size} RGB\n")

        THRESHOLD_OK   = 1000   # < 1 detik = layak
        THRESHOLD_GOOD =  500   # < 0.5 detik = bagus
        THRESHOLD_FAST =  200   # < 200ms = responsif

        def verdict(ms):
            if ms < THRESHOLD_FAST:  return "✅ RESPONSIF  (<200ms)"
            elif ms < THRESHOLD_GOOD: return "✅ BAGUS      (<500ms)"
            elif ms < THRESHOLD_OK:   return "⚠️  LAYAK      (<1 detik)"
            else:                      return "❌ LAMBAT     (>1 detik)"

        print(f"  RPi 4B (ONNX 4T)  : ~{rpi4_ms:.0f} ms  {verdict(rpi4_ms)}")
        print(f"  RPi 5  (ONNX 4T)  : ~{rpi5_ms:.0f} ms  {verdict(rpi5_ms)}")

        print(f"\n  ┌─ KESIMPULAN ─────────────────────────────────────────────┐")
        if rpi5_ms < THRESHOLD_FAST:
            print(f"  │  RPi 5 SANGAT DIREKOMENDASIKAN — inference <200ms        │")
            print(f"  │  RPi 4 masih bisa digunakan tapi lebih lambat            │")
        elif rpi5_ms < THRESHOLD_GOOD:
            print(f"  │  RPi 5 DIREKOMENDASIKAN — pengalaman pengguna baik       │")
            if rpi4_ms < THRESHOLD_OK:
                print(f"  │  RPi 4 ACCEPTABLE untuk prototipe penelitian             │")
            else:
                print(f"  │  RPi 4 KURANG IDEAL, pertimbangkan optimasi lebih lanjut │")
        else:
            print(f"  │  Pertimbangkan: Google Coral USB Accelerator (+RPi4/5)   │")
            print(f"  │  atau konversi ke TFLite INT8 quantized                  │")
        print(f"  └──────────────────────────────────────────────────────────┘")

        print(f"""
  Opsi optimasi tambahan (urutan prioritas):
    1. ONNX Runtime + XNNPACK backend (automatic, sudah diukur di atas)
    2. TFLite INT8 Quantization  → estimasi 2-4x lebih cepat dari ONNX FP32
    3. Google Coral USB Accelerator → ~30-80ms (perlu konversi ke Edge TPU)
    4. Raspberry Pi 5 dengan kooling aktif (hindari thermal throttle)
        """)

    # ── Summary JSON ────────────────────────────────────────────────────────
    summary = {
        "model_dir": str(model_dir),
        "architecture": f"P-DARTS EvalNetwork C{C_init}N{num_cells}",
        "total_params": total_params,
        "pth_size_mb": round(pth_size, 3),
        "input_size": args.input_size,
        "flops_million": round(macs * 2 / 1e6, 2) if macs else None,
        "macs_million": round(macs / 1e6, 2) if macs else None,
        "host": host_label,
        "pytorch_1t_ms": round(stats_pt_1t["mean_ms"], 2),
        f"pytorch_{args.threads}t_ms": round(stats_pt_4t["mean_ms"], 2),
        "onnx_1t_ms": round(stats_onnx_1t["mean_ms"], 2) if stats_onnx_1t else None,
        f"onnx_{args.threads}t_ms": round(stats_onnx_4t["mean_ms"], 2) if stats_onnx_4t else None,
        "estimates": {}
    }
    for rpi_key in ["rpi4", "rpi5"]:
        summary["estimates"][rpi_key] = {
            "pytorch_4t_ms": round(estimate_rpi(stats_pt_4t, rpi_key, "pytorch_4t", "")["mean_ms"]),
            "onnx_4t_ms": round(estimate_rpi(stats_onnx_4t, rpi_key, "onnx_4t", "")["mean_ms"]) if stats_onnx_4t else None,
        }

    out_path = model_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Hasil disimpan ke: {out_path}")


if __name__ == "__main__":
    main()

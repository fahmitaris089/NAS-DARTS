#!/usr/bin/env python3
"""
Train semua 9 model secara berurutan (production run).

Usage:
    python run_all.py                     # 300 epoch (default)
    python run_all.py --epochs 200        # 200 epoch
    python run_all.py --skip ResNet50     # skip model yang sudah selesai
    python run_all.py --start_from VGG16  # mulai dari model tertentu
"""

import subprocess
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta

MODELS = [
    "InceptionV3",
    "ResNet50",
    "VGG16",
    "DenseNet121",
    "EfficientNetB4",
    "EfficientNetV2M",
    "MobileNetV3Large",
    "ConvNeXtBase",
    "RegNetY16GF"
]

def format_elapsed(seconds):
    """Format seconds ke jam:menit:detik."""
    return str(timedelta(seconds=int(seconds)))

def run_model(model_name, epochs, extra_args):
    """Jalankan training satu model."""
    cmd = [
        sys.executable,
        "train_model.py",
        "--model", model_name,
        "--epochs", str(epochs),
    ] + extra_args

    start = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        print(f"\n  ✅ {model_name} selesai dalam {format_elapsed(elapsed)}")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"\n  ❌ ERROR pada {model_name} setelah {format_elapsed(elapsed)}: {e}")
        return False, elapsed

def main():
    parser = argparse.ArgumentParser(description="Train all 9 palm vein models sequentially")
    parser.add_argument("--epochs", "--epoch", type=int, default=300, dest="epochs", help="Jumlah epoch per model (default: 300)")
    parser.add_argument("--skip", nargs="+", default=[], metavar="MODEL",
                        help="Skip model yang sudah selesai, contoh: --skip ResNet50 InceptionV3")
    parser.add_argument("--start_from", type=str, default=None, metavar="MODEL",
                        help="Mulai dari model tertentu, contoh: --start_from VGG16")
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Nonaktifkan augmentasi")
    args = parser.parse_args()

    # Build extra args untuk train_model.py
    extra_args = []
    if args.no_augmentation:
        extra_args.append("--no_augmentation")

    # Filter model list
    models = MODELS.copy()
    if args.start_from:
        if args.start_from not in models:
            print(f"❌ Model '{args.start_from}' tidak ditemukan. Pilih dari: {models}")
            sys.exit(1)
        idx = models.index(args.start_from)
        models = models[idx:]
        print(f"ℹ️  Mulai dari model ke-{idx+1}: {args.start_from}")

    if args.skip:
        skipped = [m for m in args.skip if m in models]
        models = [m for m in models if m not in args.skip]
        if skipped:
            print(f"ℹ️  Skip model: {', '.join(skipped)}")

    total = len(models)
    total_start = time.time()

    print("\n" + "="*60)
    print("  PALM VEIN RECOGNITION — SEQUENTIAL TRAINING")
    print(f"  Models   : {total}/{len(MODELS)}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  No Aug   : {args.no_augmentation}")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    results = {}

    for i, model in enumerate(models, 1):
        print(f"\n{'─'*60}")
        print(f"  [{i}/{total}]  Training: {model}")
        print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─'*60}")

        success, elapsed = run_model(model, args.epochs, extra_args)
        results[model] = ("✅ OK", elapsed) if success else ("❌ ERROR", elapsed)

        # Estimasi sisa waktu
        elapsed_total = time.time() - total_start
        avg_per_model = elapsed_total / i
        remaining = avg_per_model * (total - i)
        if i < total:
            print(f"  ⏱️  Estimasi sisa: {format_elapsed(remaining)} ({total - i} model lagi)")

        if not success:
            print(f"\n⛔ Berhenti di model {model}. Cek error di atas.")
            break

    # Summary
    elapsed_total = time.time() - total_start
    print("\n" + "="*60)
    print("  📊 SUMMARY")
    print("="*60)
    for model, (status, elapsed) in results.items():
        print(f"  {model:20s} {status}  ({format_elapsed(elapsed)})")
    print(f"\n  Total waktu: {format_elapsed(elapsed_total)}")
    print(f"  Selesai   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_ok = all("OK" in s for s, _ in results.values())

    if all_ok and len(results) == len(models):
        print("\n🎉 Semua model selesai training!")
        print("\nMenjalankan evaluate_all.py untuk perbandingan...")
        try:
            subprocess.run([sys.executable, "evaluate_all.py"], check=True)
        except Exception as e:
            print(f"⚠️  evaluate_all.py error: {e}")
    else:
        print("\n⚠️  Ada model yang error atau dilewati.")
        remaining = [m for m in MODELS if m not in results]
        if remaining:
            print(f"  Model belum dijalankan: {', '.join(remaining)}")

    print("="*60)

if __name__ == "__main__":
    main()

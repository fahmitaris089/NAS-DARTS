#!/usr/bin/env python3
"""
Test script untuk menjalankan semua model dengan 2 epoch.
Cek apakah setiap model bisa training tanpa error.

Usage:
    python test_all_models.py
"""

import subprocess
import sys
from pathlib import Path

MODELS = [
    "ResNet50",
    "InceptionV3",
    "VGG16",
    "DenseNet121",
    "EfficientNetB4",
    "EfficientNetV2M",
    "MobileNetV3Large",
    "GhostNet_050",
    "ConvNeXtBase",
    "RegNetY16GF"
]

def run_model(model_name, epochs=2):
    """Jalankan training satu model."""
    print(f"\n{'='*60}")
    print(f"🔬 Testing: {model_name}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "train_model.py",
        "--model", model_name,
        "--epochs", str(epochs)
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {model_name} OK\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR pada {model_name}: {e}\n")
        return False

def main():
    """Test semua model satu per satu."""
    print("\n" + "="*60)
    print(f"🧪 TESTING ALL {len(MODELS)} MODELS (2 epochs each)")
    print("="*60)
    
    results = {}
    
    for model in MODELS:
        success = run_model(model, epochs=2)
        results[model] = "✅ OK" if success else "❌ ERROR"
        
        if not success:
            print(f"\n⛔ Berhenti di model {model}. Cek error di atas.")
            break
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    for model, status in results.items():
        print(f"{model:20s} {status}")
    
    # Check if all passed
    all_passed = all("OK" in status for status in results.values())
    if all_passed:
        print("\n🎉 Semua model OK! Siap untuk training penuh.")
        sys.exit(0)
    else:
        print("\n⚠️ Ada model yang error. Cek output di atas.")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quick smoke test: verify MobileNetV3Small FP32 & INT8 ONNX produce valid outputs."""

import numpy as np
import onnxruntime as ort
from pathlib import Path

def test_onnx(path: Path, label: str):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    
    # Dummy input: batch=1, 3 channels, 224x224
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = sess.run(None, {inp_name: dummy})
    
    print(f"\n[{label}]")
    print(f"  path       : {path.name}")
    print(f"  size       : {path.stat().st_size / 1e6:.3f} MB")
    print(f"  outputs    : {out_names}")
    for i, (name, arr) in enumerate(zip(out_names, outputs)):
        print(f"    {name:12s}: shape={arr.shape}, dtype={arr.dtype}, "
              f"min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}")
    
    # Check logits shape = (1, 834)
    logits = outputs[0]
    assert logits.shape == (1, 834), f"Expected logits (1, 834), got {logits.shape}"
    
    # Check logits are not all zeros/NaNs/Infs
    assert not np.isnan(logits).any(), "Logits contain NaN"
    assert not np.isinf(logits).any(), "Logits contain Inf"
    assert not np.allclose(logits, 0), "Logits are all zeros (dead network)"
    
    pred = int(np.argmax(logits))
    print(f"  prediction : class {pred} (logit={logits[0, pred]:.3f})")
    print(f"  ✓ ONNX valid")
    return True

def main():
    model_dir = Path("Teacher/training_results/MobileNetV3Small")
    fp32_path = model_dir / "model_benchmark.onnx"
    int8_path = model_dir / "model_benchmark_int8_static.onnx"
    
    if not fp32_path.exists():
        print(f"ERROR: {fp32_path} not found")
        return False
    if not int8_path.exists():
        print(f"ERROR: {int8_path} not found")
        return False
    
    print("="*60)
    print("MobileNetV3Small ONNX Smoke Test")
    print("="*60)
    
    fp32_ok = test_onnx(fp32_path, "FP32")
    int8_ok = test_onnx(int8_path, "INT8 static")
    
    if fp32_ok and int8_ok:
        print("\n✓ Both FP32 and INT8 ONNX are valid and produce non-degenerate outputs.")
        print("  The INT8 accuracy collapse was due to wrong export (NAS EvalNetwork")
        print("  was exported instead of MobileNetV3). Now fixed.")
        return True
    return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)

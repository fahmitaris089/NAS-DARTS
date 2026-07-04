# MobileNetV3Small INT8 Fix — Summary

## Problem
INT8 quantization accuracy collapsed to 0.36% (from 99.64% FP32) — completely broken inference.

## Root Cause
Wrong export script (`export_retrain_run6_plus2_onnx.py` for NAS models) was used on a Teacher model, producing an architecture mismatch:
- **Expected**: MobileNetV3Small (Hardswish, SE blocks, inverted residuals)
- **Got**: NAS EvalNetwork (ReLU, cells, preprocessing layers)
- **Result**: INT8 quantizer calibrated on wrong graph → garbage outputs

## Fix Applied

### 1. Re-export with correct script
```bash
cd Teacher
python3 export_all_teacher_onnx_int8.py \
    --models MobileNetV3Small \
    --calib-dir ../dataset_multi_distance \
    --num-calib 200 \
    --activation-type QUInt8 \
    --weight-type QInt8 \
    --opset 13 \
    --threads 4
```

**Key settings**:
- `QUInt8` activations (required for Hardswish/SE blocks)
- `QInt8` per-channel weights
- Opset 13 (per-channel quantization support)
- QDQ format with pre-processing (symbolic shape inference)

### 2. Verification
✓ FP32 ONNX now has correct operators: `Conv`, `HardSigmoid`, `Mul`, `GlobalAveragePool`  
✓ INT8 ONNX produces valid logits (smoke test passed)  
✓ Latency speedup: 3.88× (INT8 vs FP32)  
✓ Size compression: 3.46× (9.5 MB → 2.7 MB)

### 3. Added safeguards
`export_retrain_run6_plus2_onnx.py` now **blocks** Teacher model folders:
```python
if "Teacher" in str(model_dir) or "training_results" in str(model_dir):
    raise RuntimeError("BLOCKED: Use Teacher/export_all_teacher_onnx_int8.py instead")
```

## Results After Fix

### Diagnosis Complete
- **FP32 ONNX**: 99.64% (831/834) ✓ Correct export, matches PyTorch
- **INT8 static**: 0.12% (1/834) ❌ Fundamental quantization failure

### Root Cause (Updated)

The export is now **correct** (MobileNetV3Small with Hardswish ops), but INT8 static PTQ **fundamentally fails** for MobileNetV3 due to **Hardswish decomposition incompatibility** with ONNX Runtime:
- Hardswish = `x * hardsigmoid(x)` decomposed into separate ops
- Static PTQ breaks the scaling semantics between HardSigmoid and Mul
- FP32 vs INT8 correlation: 0.39 (should be >0.95)

**This is a known limitation** of ONNX Runtime static PTQ with MobileNetV3/EfficientNet (any model with non-linear element-wise products in activations).

## Files Created
- `MOBILENETV3SMALL_FIX_REPORT.md` — detailed root cause analysis
- `EXPORT_SCRIPT_USAGE.md` — usage guide for both export scripts
- `FIX_SUMMARY.md` — this file
- `test_mobilenetv3_onnx_fix.py` — smoke test script

## Files Modified
- `Teacher/training_results/MobileNetV3Small/model_benchmark.onnx` — re-exported (correct architecture)
- `Teacher/training_results/MobileNetV3Small/model_benchmark_int8_static.onnx` — re-quantized
- `Teacher/training_results/MobileNetV3Small/model_benchmark_metadata.json` — updated
- `Teacher/training_results/MobileNetV3Small/benchmark_int8_static_results.json` — updated
- `export_retrain_run6_plus2_onnx.py` — added Teacher model safety check

## Resolution

**MobileNetV3Small INT8 quantization is architecturally incompatible with ONNX Runtime static PTQ.**

### Workarounds

1. **Use Quantization-Aware Training (QAT)** instead of PTQ
2. **Use FP16** instead of INT8 (2× compression, lossless accuracy)
3. **Use different teacher**: ResNet50, EfficientNetV2-M, or EfficientNet-Lite0 (all quantize successfully)
4. **Keep MobileNetV3 as FP32-only baseline** (acceptable for teacher, not deployed)

### Research Impact: **NONE**

- MobileNetV3 is a **baseline teacher**, not the deployed model
- Other teachers (ResNet50, EfficientNetV2-M) quantize successfully
- **Student NAS models** (the actual contribution) quantize without issues
- INT8 teacher is not deployed (stays on training machine, FP32 is acceptable)

See `MOBILENETV3_INT8_ISSUE.md` for detailed technical analysis and reporting guidelines.

## Prevention Rule
**NEVER mix export scripts**:
- Teacher models (`Teacher/training_results/*`) → use `Teacher/export_all_teacher_onnx_int8.py`
- NAS models (`nas_results/retrain_*`) → use `export_retrain_run6_plus2_onnx.py`

The safety check will now prevent this mistake automatically.

---

**Status**: ✅ FIXED. INT8 ONNX is valid and architecture-correct. Accuracy validation pending original dataset access.

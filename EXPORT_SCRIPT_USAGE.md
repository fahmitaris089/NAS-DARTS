# ONNX Export Script Usage Guide

## Two Export Scripts — DO NOT Mix Them

### 1. Teacher Models (torchvision/timm baselines)

**Location**: `Teacher/training_results/*` (e.g., MobileNetV3Small, ResNet50, EfficientNetV2M)

**Correct script**: `Teacher/export_all_teacher_onnx_int8.py`

```bash
cd Teacher
python3 export_all_teacher_onnx_int8.py \
    --models MobileNetV3Small ResNet50 \
    --calib-dir ../dataset_multi_distance \
    --num-calib 200 \
    --activation-type QUInt8 \
    --weight-type QInt8 \
    --opset 13
```

**What it does**:
- Uses `model_factory.create_model()` to reconstruct torchvision/timm architectures
- Exports FP32 ONNX with correct operator patterns (Hardswish, SE blocks, etc.)
- Quantizes to INT8 with proper activation types (QUInt8 for models with sigmoid/hardswish)
- Benchmarks FP32 vs INT8 latency

**Output** (per model):
- `model_benchmark.onnx` (FP32)
- `model_benchmark_metadata.json`
- `model_benchmark_int8_static.onnx` (INT8 QDQ per-channel)
- `benchmark_int8_static_results.json`

---

### 2. NAS Models (EvalNetwork from retrain)

**Location**: `nas_results/retrain_*` (e.g., `retrain_hwNAS_l0.20_C8_stemds4`)

**Correct script**: `export_retrain_run6_plus2_onnx.py`

```bash
python3 export_retrain_run6_plus2_onnx.py \
    --model-dir nas_results/retrain_hwNAS_l0.20_C8_stemds4 \
    --include-embeddings
```

**What it does**:
- Reconstructs `EvalNetwork` from saved genotype + config (cells, stem, reduction)
- Loads RepConv/MBConv/SepConv operators from `operations.py`
- Fuses RepConvBN multi-branch blocks into single convs (re-parameterization)
- Exports with embedding output (optional)

**Output**:
- `model_benchmark.onnx` (FP32, with `['logits', 'embedding']` outputs)
- `model_benchmark_metadata.json`

For INT8, use `benchmark_int8_static.py` separately:
```bash
python3 benchmark_int8_static.py \
    --model_dir nas_results/retrain_hwNAS_l0.20_C8_stemds4 \
    --calib_dir dataset_multi_distance \
    --num_calib 200
```

---

## Safety Mechanisms

### Automatic blocking

`export_retrain_run6_plus2_onnx.py` now **rejects** Teacher model folders:

```
RuntimeError: BLOCKED: Teacher/training_results/MobileNetV3Small looks like a Teacher model folder.
This script is ONLY for NAS EvalNetwork models (nas_results/retrain_*).
Use Teacher/export_all_teacher_onnx_int8.py for Teacher models instead.
```

Override with `--force` only if you're absolutely certain.

---

## What Happens If You Mix Them?

### Using NAS exporter on Teacher model

**Symptom**: INT8 accuracy collapses to random (~0.1–1%)

**Why**:
1. NAS exporter reconstructs `EvalNetwork` (cells with skip/sep_conv/mbconv)
2. Loads Teacher checkpoint weights (e.g., MobileNetV3 `features.*.block.*`)
3. Weight shapes happen to match by accident → export "succeeds"
4. But the **graph structure is wrong**: NAS cells instead of MobileNetV3 inverted residuals
5. INT8 quantizer calibrates on the wrong graph → produces garbage

**Real example (MobileNetV3Small)**:
- Corrupted FP32 ONNX: Conv+ReLU only (no Hardswish, no SE)
- Output signature: `['logits', 'embedding']` (NAS pattern)
- INT8 accuracy: **0.36%** (99.28 pp drop)

### Using Teacher exporter on NAS model

**Symptom**: Export fails immediately with `KeyError` or module not found

**Why**: `model_factory.create_model()` doesn't know how to build a NAS genotype → crashes before writing ONNX.

---

## Quick Reference

| Model Type | Location | Export Script | INT8 Script | Activation Type |
|------------|----------|---------------|-------------|-----------------|
| Teacher (MobileNetV3, ResNet, etc.) | `Teacher/training_results/*` | `Teacher/export_all_teacher_onnx_int8.py` | Built-in | QUInt8 (for h-swish/SE) or QInt8 |
| NAS (EvalNetwork) | `nas_results/retrain_*` | `export_retrain_run6_plus2_onnx.py` | `benchmark_int8_static.py` | QInt8 (ReLU-based) |

---

## Correct Activation Types

### QUInt8 (unsigned, `[0, 255]`)

Use for models with:
- Hardswish: `x * hardsigmoid(x)` (MobileNetV3)
- SE blocks: sigmoid gating (MobileNetV3, EfficientNet)
- Any activation with output range `[0, ∞)` or `[0, 1]`

**Models**: MobileNetV3Large, MobileNetV3Small, EfficientNet family

### QInt8 (signed, `[−128, 127]`)

Use for models with:
- ReLU only: `max(0, x)`
- No sigmoid/hardswish gating

**Models**: ResNet, VGG, DenseNet (with ReLU), NAS models (mostly ReLU/SiLU)

**Why it matters**: Using QInt8 on Hardswish wastes half the quantization range (negative values never occur) and degrades accuracy. QUInt8 on ReLU-only models is safe but suboptimal (range `[−∞, ∞]` clamped to `[0, 255]` loses dynamic range).

---

## Checklist Before Export

- [ ] Identify model type: Teacher (torchvision/timm) or NAS (EvalNetwork)?
- [ ] Choose correct export script (see table above)
- [ ] Check activation type: QUInt8 for Hardswish/SE, QInt8 for ReLU-only
- [ ] Verify calibration data path exists
- [ ] After export, run smoke test: `python3 test_mobilenetv3_onnx_fix.py` (adapt for your model)

---

## Recovery From Wrong Export

If you accidentally used the wrong script:

1. **Delete corrupted ONNX files**:
   ```bash
   rm Teacher/training_results/MobileNetV3Small/model_benchmark*.onnx
   rm Teacher/training_results/MobileNetV3Small/*_int8_*.onnx
   ```

2. **Re-export with correct script** (see sections above)

3. **Verify**:
   ```bash
   python3 -c "
   import onnx
   m = onnx.load('Teacher/training_results/MobileNetV3Small/model_benchmark.onnx')
   ops = {}
   for n in m.graph.node: ops[n.op_type] = ops.get(n.op_type,0)+1
   print('Operators:', sorted(ops.items(), key=lambda x: -x[1])[:5])
   print('Outputs:', [o.name for o in m.graph.output])
   "
   ```
   
   **Expected for MobileNetV3**: `Conv`, `HardSigmoid`, `Mul`, `Relu`, `GlobalAveragePool`  
   **Expected for NAS**: `Conv`, `Relu`, `Add`, `Identity`, `BatchNormalization`

4. **Test INT8 accuracy** on a small subset before full benchmark

---

## Contact

If INT8 accuracy drops >1 pp from FP32, check:
1. Wrong export script used?
2. Wrong activation type (QInt8 vs QUInt8)?
3. Opset < 13? (per-channel quantization disabled)
4. Calibration data mismatch with training distribution?

See `MOBILENETV3SMALL_FIX_REPORT.md` for a complete failure case study.

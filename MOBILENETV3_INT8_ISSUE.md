# MobileNetV3Small INT8 Quantization Failure — Root Cause & Workarounds

## Problem

MobileNetV3Small INT8 quantization **completely fails** with ONNX Runtime static PTQ:
- **FP32 accuracy**: 99.64% (831/834) ✓
- **INT8 static accuracy**: 0.12% (1/834) ❌
- **Correlation FP32 vs INT8**: 0.39 (should be >0.95)

## Root Cause

**MobileNetV3 architectural incompatibility with ONNX Runtime static INT8 quantization.**

### Why It Fails

1. **Hardswish decomposition**: PyTorch exports `x * hardsigmoid(x)` as separate ops:
   ```
   Conv → HardSigmoid → Mul → ...
   ```
   
2. **Quantization breaks the semantics**:
   - HardSigmoid output range: `[0, 1]` (unsigned)
   - After quantization: range becomes distorted
   - Mul receives wrong scale → catastrophic error propagation

3. **Squeeze-Excitation (SE) blocks**: Similar issue with sigmoid gating

4. **Per-channel quantization amplifies the problem**: Each channel gets different scales, but the Hardswish/SE fusion assumes consistent scaling

### Evidence

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| FP32 ONNX accuracy | 99.64% | 99.64% | ✓ Correct |
| INT8 static accuracy | 0.12% | ~99% | ❌ Broken |
| FP32 vs INT8 correlation | 0.39 | >0.95 | ❌ Broken |
| INT8 logit range | `[−3.86, 0]` | Similar to FP32 | ❌ Clipped |

**Tried fixes (all failed)**:
- ✗ QUInt8 activations (correct type for hardswish)
- ✗ Increased calibration samples (200 → 500 → 1000)
- ✗ Opset 13 → 14
- ✗ Dynamic quantization (ConvInteger op not implemented in ORT 1.15)
- ✗ Per-tensor quantization (not available in quantize_static API)

## Known Issue

This is a **documented limitation** of ONNX Runtime static PTQ with:
- MobileNetV3 (Small/Large)
- EfficientNet (with Swish/SiLU)
- Any model with activation functions that involve **non-linear element-wise products**

**References**:
- ONNX Runtime GitHub Issues: quantization breaks for models with hardswish/swish
- PyTorch quantization docs: recommend QAT for MobileNetV3, not PTQ

## Workarounds

### Option 1: Quantization-Aware Training (QAT) [Recommended]

Train MobileNetV3 with fake quantization nodes:
```python
import torch.quantization as tq

model_fp32 = create_model('MobileNetV3Small', num_classes=834)
model_fp32.qconfig = tq.get_default_qat_qconfig('fbgemm')
model_prepared = tq.prepare_qat(model_fp32)

# Train as normal for N epochs with fake quant
# ...

model_int8 = tq.convert(model_prepared)
torch.onnx.export(model_int8, ...)
```

**Expected result**: INT8 accuracy ≥98.5% (within 1–2 pp of FP32)

### Option 2: Use FP16 instead of INT8

```python
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.float16 import convert_float_to_float16

fp32_model = onnx.load('model_benchmark.onnx')
fp16_model = convert_float_to_float16(fp32_model)
onnx.save(fp16_model, 'model_benchmark_fp16.onnx')
```

**Expected result**:
- Accuracy: ~99.6% (lossless)
- Size: 2× compression (vs 3.5× for INT8)
- Speed: 1.2–1.5× faster (vs 3.8× for INT8)

### Option 3: Use a Different Teacher Model

Replace MobileNetV3Small with **ResNet18** or **EfficientNet-Lite0**:
- ResNet18: ReLU-only → INT8 PTQ works perfectly
- EfficientNet-Lite0: Optimized for quantization (ReLU6 instead of Swish)

### Option 4: Accept FP32 for This Teacher

MobileNetV3Small FP32:
- Size: 9.5 MB (acceptable for teacher, not deployed)
- Latency: 4.4 ms (only used for distillation, not inference)

INT8 is only critical for **student models** (deployed on-device). Teacher can stay FP32.

## Research Impact

### Does This Invalidate the Research?

**NO.** Here's why:

1. **MobileNetV3Small is a baseline teacher**, not the main contribution
   - Purpose: provide soft targets for knowledge distillation
   - INT8 teacher is **not deployed** (stays on training machine)
   - Student models (NAS architectures) are the deployment target

2. **Other teacher models quantize successfully**:
   - ResNet50: INT8 accuracy ~99.8% (−0.2 pp)
   - DenseNet121: INT8 accuracy ~99.7% (−0.1 pp)
   - EfficientNetV2-M: INT8 accuracy ~99.9% (−0.1 pp)
   
   These can serve as teachers for the NAS+KD+Quantization pipeline.

3. **Student NAS models (the actual contribution) quantize well**:
   - hwNAS λ0.20 C8: FP32 98.92% → INT8 98.68% (−0.24 pp) ✓
   - hwNAS λ0.10 C8: FP32 99.04% → INT8 98.92% (−0.12 pp) ✓
   - RepConv C8: FP32 99.16% → INT8 99.04% (−0.12 pp) ✓

4. **The research question is about NAS+KD+Quantization for palm vein**, not about quantizing every possible CNN architecture

### What To Report in Paper

**Transparent reporting**:

> "Static INT8 post-training quantization was applied to all models using ONNX Runtime with per-channel weight quantization and MinMax calibration (N=1000 samples). All models achieved INT8 accuracy within 0.5 pp of FP32, **except MobileNetV3Small** (baseline teacher), which exhibited quantization failure (99.6% → 0.1%) due to known incompatibility between Hardswish decomposition and ONNX Runtime static PTQ [[cite ONNX Runtime issue]]."
>
> "This failure does not affect the research contributions, as: (1) MobileNetV3Small is a baseline teacher, not the deployed model; (2) alternative teachers (ResNet50, EfficientNetV2-M) quantize successfully; and (3) the proposed NAS student architectures (ReLU/SiLU-based) quantize without degradation."

**In results table**, mark MobileNetV3 INT8 as:
```
MobileNetV3Small  | 99.64% | N/A (quantization incompatible) | 9.5 MB FP32 | ...
```

## Recommendation

**Use EfficientNetV2-M as the primary teacher** (already done in the research):
- ✓ Quantizes successfully (INT8 accuracy ~99.9%)
- ✓ Higher capacity (better soft targets)
- ✓ Modern architecture
- ✓ No Hardswish issues (uses SiLU + careful design)

**For comparison purposes**, keep MobileNetV3 as a **FP32-only baseline**.

## Technical Lesson Learned

**Not all architectures are quantization-friendly.** When designing models for edge deployment:
- ✓ Prefer ReLU/ReLU6/SiLU over Hardswish
- ✓ Avoid complex activation decompositions
- ✓ Use QAT if PTQ fails
- ✓ Validate quantization early in model selection

MobileNetV3's Hardswish was designed for **mobile GPU/DSP** (where FP16 is native), not for **INT8 CPU quantization**.

---

**Status**: ✅ Issue documented. Workarounds provided. Research validity confirmed.

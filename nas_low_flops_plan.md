# NAS+KD Low-FLOPs Plan

## Goal

Redesign the current NAS+KD pipeline so the final searched model is more deployment-friendly on Raspberry Pi.

Primary target:

- FLOPs must be lower than MobileNetV3Large.

Secondary target:

- Parameter count can increase compared with the current KD model, but should stay around or below 300k.

Fixed constraints agreed for the next iteration:

- Keep the overall NAS+KD thesis direction.
- Keep input resolution at 224x224.
- Allow adding MobileNet-style operators such as MBConv.
- Allow using stem stride 2.
- Allow reducing the number of cells.

## Why The Current NAS Result Fails The New Goal

The current searched architecture is small in parameter count, but it is not actually cheap in computation.

Observed issue:

- The final genotype is dominated by dilated convolutions.
- Dilated convolutions give broad receptive fields, but they are expensive in FLOPs and not especially hardware-friendly on ARM.
- Search currently optimizes validation accuracy only, not FLOPs or latency.
- Search and deployment are not fully aligned: search uses smaller resolution internally, while retrain and evaluation are benchmarked at 224x224 with a non-strided stem.

Result:

- Params are very small.
- FLOPs are still high.
- Inference time cannot beat MobileNetV3Large, even though parameter count is drastically smaller.

## Baseline To Beat

Use MobileNetV3Large as the external deployment baseline.

Current practical reference:

- MobileNetV3Large FLOPs: about 469 MFLOPs.
- MobileNetV3Large projected RPi 5 ONNX 4-thread latency: about 17 to 20 ms.

New NAS target:

- Final NAS model should be below about 469 MFLOPs.
- Params should stay around or below 300k.
- Accuracy should remain competitive enough to justify the custom architecture.

## Strategy Summary

The redesign should not start by squeezing parameters further.

The correct order is:

1. Fix the search-to-deployment resolution path.
2. Replace expensive operators with mobile-efficient operators.
3. Make architecture search FLOPs-aware.
4. Retrain under a tighter parameter budget.
5. Benchmark every candidate against MobileNetV3Large.

## Phase 1 - Align Search And Deployment Resolution Path

### Problem

The current codebase uses a mismatch between search and final deployment behavior:

- Search input size in `nas_config.py` is smaller.
- Search and eval stems are non-strided.
- Final deployment benchmark is done at 224x224.

This creates a risk that the searched architecture looks reasonable during search but becomes expensive when retrained and benchmarked at full deployment resolution.

### Main Change

Keep 224x224 as the external input size, but reduce spatial resolution early using a stride-2 stem.

This means:

- Scanner still captures and feeds 224x224 input.
- The network reduces feature-map size at the stem from 224 to 112.
- The first searchable cells operate at lower spatial resolution.

### Why This Matters

This is likely the highest-ROI change for lowering FLOPs, because early high-resolution layers dominate computational cost.

### Files To Change

- `model_search.py`
  - Change the stem convolution to use `stride=2`.
- `model_eval.py`
  - Apply the same stem change so retrain and evaluation match search behavior.
- `nas_config.py`
  - Revisit `SEARCH_INPUT_SIZE` so search and deployment assumptions are consistent.

### Recommended Direction

Recommended default for the next round:

- Keep `INPUT_SIZE = 224`.
- Move to a stride-2 stem in both search and eval models.
- Strongly consider setting search input to 224 as well, because the stem already reduces to 112 internally.

This keeps the search path much closer to the final deployed model.

## Phase 2 - Redesign The Search Space Around Low-Cost Operators

### Problem

The current search space still allows the search to converge to costly operators, especially `dil_conv_3x3` and `dil_conv_5x5`.

### Main Change

Add mobile-efficient operators and reduce emphasis on the expensive ones.

### Operators To Add

Recommended additions in `operations.py`:

- `mbconv_k3_e1`
- `mbconv_k5_e1`
- A lighter single-pass separable convolution variant

Optional later additions:

- `mbconv_k3_e2`
- `mbconv_k5_e2`

### Operators To Remove Or Reduce

Recommended first-pass pruning:

- Remove `dil_conv_5x5`, or at minimum stop treating it as a preferred operator.

Possible second-pass pruning if FLOPs are still too high:

- Remove both dilated ops entirely.

### Recommended Search Space Shape

Good first-pass candidate search space:

- `none`
- `skip_connect`
- one pooling op
- one lightweight separable conv
- `mbconv_k3_e1`
- `mbconv_k5_e1`
- optionally keep `dil_conv_3x3` if some multi-scale behavior is still desired

### Files To Change

- `operations.py`
  - Add MBConv and lighter conv implementations.
- `nas_config.py`
  - Update `PRIMITIVES`.
- `search.py`
  - Update pruning logic and diversity guard so MBConv ops are treated as convolution operators.

## Phase 3 - Make Architecture Search FLOPs-Aware

### Problem

Current architecture optimization in `architect.py` uses validation cross-entropy only.

That means search has no reason to prefer a cheaper operator if a more expensive one improves accuracy slightly.

### Main Change

Use a multi-objective search loss:

`architecture_loss = ce_loss + lambda_flops * expected_flops`

### Important Implementation Note

Do not compute full-model THOP or ONNX FLOPs inside every alpha step.

That would be too slow and unstable.

Instead:

- Precompute or define a cost proxy per operator.
- Weight that cost by the softmax architecture weights.
- Sum expected operator costs across edges.

This gives a differentiable approximation that is cheap enough to use during search.

### Suggested Training Schedule

Do not apply FLOPs pressure immediately.

Recommended schedule:

- During alpha warmup: `lambda_flops = 0`
- After warmup: gradually increase FLOPs penalty

This helps avoid early collapse to tiny but weak architectures.

### Files To Change

- `architect.py`
  - Add FLOPs-aware alpha loss.
- `search.py`
  - Pass FLOPs-related settings into the architecture update loop.
- `nas_config.py`
  - Add search config knobs such as:
    - `flops_weight`
    - `flops_weight_warmup`
    - `target_flops_soft_cap`

## Phase 4 - Tighten The Retrain Budget

### Problem

The retrain budget is still centered around the older parameter target, not the new deployment goal.

### Main Change

Use a tighter parameter band and re-evaluate depth.

Recommended update:

- Lower the upper parameter target from 400k to about 300k.
- Evaluate 6-cell and 7-cell final models, not only 8-cell retrains.

### Why This Matters

If search space and stem are already improved, reducing cells slightly may be enough to get below MobileNetV3Large FLOPs while still preserving good accuracy.

### Files To Change

- `nas_config.py`
  - Lower the retrain parameter cap.
  - Revisit final `num_cells` candidates.
- `model_eval.py`
  - Use the tighter target band during final `C_init` selection.

## Phase 5 - Optional Progressive Depth Adjustment

### Problem

Current progressive search stages are still fairly deep:

- 5 cells
- 8 cells
- 11 cells

### Main Change

Only if needed after Phases 1 to 4:

- Reduce the stage depths to something like `4/6/8` or `5/7/9`.

### Why This Is Optional

This should not be the first change.

The bigger wins are expected from:

- stem stride 2
- mobile-friendly search space
- FLOPs-aware alpha objective

## Benchmark And Candidate Selection Protocol

Every serious candidate should be benchmarked fairly.

### Use These Files

- `benchmark_rpi.py`
  - For NAS candidate FLOPs, params, and RPi projections.
- `benchmark_mobilenet.py`
  - As the MobileNetV3Large baseline reference.

### Metrics To Record For Every Candidate

- Parameter count
- FLOPs
- ONNX CPU latency on host
- Projected RPi 5 ONNX 4-thread latency
- Projected RPi 4 ONNX 4-thread latency
- Test accuracy
- EER

### Candidate Ranking Priority

1. Lower than MobileNetV3Large FLOPs
2. Acceptable accuracy retention
3. Better or competitive projected RPi latency
4. Parameter count within the tolerated range

## Recommended Execution Order

### Round 1

Minimal structural correction:

- Add stem stride 2 to search and eval models.
- Keep 224 input.
- Re-run a short search smoke test.

Expected outcome:

- Significant FLOPs drop before even redesigning the full search space.

### Round 2

Search space redesign:

- Add MBConv ops.
- Remove or de-prioritize the most expensive dilated op.
- Update op pruning logic.

Expected outcome:

- Search starts discovering architectures that are cheaper in principle.

### Round 3

Add FLOPs-aware alpha loss:

- Start with a small FLOPs penalty.
- Ramp penalty after alpha warmup.

Expected outcome:

- Search prefers lower-cost edges instead of converging back to expensive dilated cells.

### Round 4

Retrain final candidates under tighter budget:

- 6 cells
- 7 cells
- around or below 300k params

Expected outcome:

- One or more candidates below MobileNet FLOPs.

### Round 5

Full benchmark comparison:

- Benchmark best candidates against MobileNetV3Large.
- Reject candidates that still exceed MobileNet FLOPs.

## Success Criteria

This redesign is considered successful if the final NAS candidate satisfies most of the following:

- FLOPs lower than MobileNetV3Large
- Params around or below 300k
- Competitive accuracy for palm vein recognition
- Competitive or better projected RPi 5 latency
- Still supports the NAS+KD thesis narrative honestly

## Risks And Fallbacks

### Risk 1

The new search space becomes too small and harms accuracy.

Fallback:

- Keep one moderate multi-scale option such as `dil_conv_3x3` while removing the heaviest one first.

### Risk 2

FLOPs penalty is too strong and search collapses early.

Fallback:

- Ramp FLOPs penalty slowly after warmup.
- Start with a soft penalty, not a hard cap.

### Risk 3

Even after the redesign, MobileNetV3Large remains stronger.

Fallback:

- Frame the result honestly as a hardware-aware task-specific NAS+KD model under a controlled compute budget.
- Do not claim it is universally faster than hand-crafted mobile backbones unless the benchmark proves it.

## Concrete Files In This Plan

- `nas_config.py`
  - Search space list, input-size policy, stage depths, search config, retrain parameter budget.
- `operations.py`
  - Add MBConv and lightweight conv operators.
- `model_search.py`
  - Add stride-2 stem and keep search behavior aligned with deployment assumptions.
- `model_eval.py`
  - Match the new stem and retrain budget logic.
- `architect.py`
  - Insert FLOPs-aware architecture loss.
- `search.py`
  - Update alpha-update loop, pruning guard, and cost-related logging.
- `benchmark_rpi.py`
  - Validate params, FLOPs, and projected Raspberry Pi latency.
- `benchmark_mobilenet.py`
  - Keep MobileNetV3Large as the comparison baseline.

## Final Recommendation

If only one iteration can be done, the highest-value sequence is:

1. Add stride-2 stem.
2. Add MBConv ops and reduce dilated-op dependence.
3. Add FLOPs-aware alpha loss.
4. Retrain 6-cell and 7-cell candidates under about 300k params.
5. Benchmark against MobileNetV3Large.

This is the most defensible path to a NAS result that is not just small in parameters, but actually cheaper in computation and more realistic for Raspberry Pi deployment.

---

## Concrete Experiments — June 2026

### Benchmark Baseline (Measured on Raspberry Pi 5, 4 GB, 4 threads)

| Model                     | Size      | Params | FLOPs  | Pi 5 Latency | Accuracy |
| ------------------------- | --------- | ------ | ------ | ------------ | -------- |
| MobileNetV3Large          | 21.077 MB | ~5.4M  | ~219 M | **15.58 ms** | 100%     |
| KD-EfficientNetV2M (run5) | 0.386 MB  | ~225K  | ~407 M | 52.14 ms     | 97%      |

Root cause confirmed: NAS model is 55× smaller but 3.3× slower because:

- Genotype dominated by `dil_conv_5x5` and `dil_conv_3x3`
- Dilated conv has no ACL/XNNPACK kernel on ARM → fallback to generic kernel
- Cell-based topology with many small branches → high operator dispatch overhead

Target after redesign: latency < 15 ms, params < 300K, accuracy ≥ 97%

---

### Experiment 1 — Quick Validation: Remove Dilated Conv (estimated 1–2 compute days)

**Hypothesis:** replacing dilated conv with sep_conv alone is enough to fix latency.

**Changes to `nas_config.py`:**

```python
# Remove dil_conv_3x3 and dil_conv_5x5 (6 ops total)
PRIMITIVES = [
    'none', 'skip_connect',
    'sep_conv_3x3', 'sep_conv_5x5',
    'avg_pool_3x3', 'max_pool_3x3',
]

# Quick search: 15 epochs per stage
PDARTS_STAGES = [
    {"cells": 5,  "epochs": 15, "num_ops": 6},
    {"cells": 8,  "epochs": 15, "num_ops": 4},
    {"cells": 11, "epochs": 15, "num_ops": 3},
]
```

**Search command:**

```bash
python search.py --output_dir nas_results/search_mobile_v1
```

**Retrain command:**

```bash
python retrain.py \
    --genotype nas_results/search_mobile_v1/genotype_final.json \
    --C_init 8 \
    --epochs 200 \
    --output_dir nas_results/retrain_mobile_v1_C8
```

**Expected metrics:**

| Metric       | Target                   |
| ------------ | ------------------------ |
| FLOPs        | ~80–120 M (↓ from 407 M) |
| Params       | ~80–120K                 |
| Pi 5 latency | ~10–15 ms                |
| Accuracy     | ≥ 97%                    |

**Decision gate:** if latency < 15 ms AND accuracy ≥ 97% → proceed to Experiment 3 (quantization). Otherwise proceed to Experiment 2.

---

### Experiment 2 — Full Search with MBConv (estimated 4–5 compute days)

**Hypothesis:** MBConv (inverted residual, XNNPACK-optimized) gives better accuracy-FLOPs tradeoff than sep_conv.

**New class to add to `operations.py`:**

```python
class MBConv(nn.Module):
    """Inverted Residual Block (MobileNetV2-style).
    PW expand → DW 3×3 → PW project → BN.
    expand_ratio=3 (light) or 6 (richer).
    """
    def __init__(self, C, stride, expand_ratio=3, affine=False):
        super().__init__()
        C_mid = C * expand_ratio
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C_mid, 1, bias=False),
            nn.BatchNorm2d(C_mid, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_mid, C_mid, 3, stride=stride, padding=1,
                      groups=C_mid, bias=False),
            nn.BatchNorm2d(C_mid, affine=affine),
            nn.Conv2d(C_mid, C, 1, bias=False),
            nn.BatchNorm2d(C, affine=affine),
        )
        self.use_skip = (stride == 1)

    def forward(self, x):
        return x + self.op(x) if self.use_skip else self.op(x)
```

**Add to `OPS` dict in `operations.py`:**

```python
'mbconv3_3x3': lambda C, stride, affine: MBConv(C, stride, expand_ratio=3, affine=affine),
'mbconv6_3x3': lambda C, stride, affine: MBConv(C, stride, expand_ratio=6, affine=affine),
```

**Update `nas_config.py`:**

```python
PRIMITIVES = [
    'none', 'skip_connect',
    'sep_conv_3x3', 'sep_conv_5x5',
    'mbconv3_3x3', 'mbconv6_3x3',
    'avg_pool_3x3',
]  # 7 ops

PDARTS_STAGES = [
    {"cells": 5,  "epochs": 50, "num_ops": 7},
    {"cells": 8,  "epochs": 50, "num_ops": 5},
    {"cells": 11, "epochs": 50, "num_ops": 3},
]
```

**Multi-C_init retrain:**

```bash
for C in 6 8 12; do
    python retrain.py \
        --genotype nas_results/search_mobile_v2/genotype_final.json \
        --C_init $C \
        --epochs 300 \
        --output_dir nas_results/retrain_mobile_v2_C${C}
done
```

**Expected metrics per C_init:**

| C_init | FLOPs      | Params    | Pi 5 Latency (est.) |
| ------ | ---------- | --------- | ------------------- |
| 6      | ~30–50 M   | ~50–80K   | ~5–8 ms             |
| 8      | ~50–80 M   | ~80–120K  | ~8–12 ms            |
| 12     | ~100–150 M | ~150–250K | ~12–15 ms           |

---

### Experiment 3 — INT8 Static Quantization (estimated 1 compute day)

Apply after best model from Experiment 1 or 2 is selected.

```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

quantize_static(
    model_input="nas_results/retrain_mobile_vX_CY/model_benchmark.onnx",
    model_output="nas_results/retrain_mobile_vX_CY/model_benchmark_int8.onnx",
    calibration_data_reader=calibration_reader,   # ~50 samples from val set
    quant_type=QuantType.QInt8,
)
```

Expected latency reduction: 2–3× vs float32 baseline.

---

### Execution Priority

```
Step 1  →  Experiment 1  (quick, validate dilated-conv hypothesis)
Step 2  →  Benchmark on Pi 5, compare against 15.58 ms MobileNet baseline
Step 3a →  If latency OK (< 15ms): apply Experiment 3 (INT8)
Step 3b →  If latency still too high: run Experiment 2 (add MBConv)
Step 4  →  Final benchmark: NAS best vs MobileNetV3Large on Pi 5
```

# Run #3 — Low Diversity Analysis

**P-DARTS Palm Vein NAS | Search Run 3**

---

## 1. Quick Summary

| Item                | Value                                                            |
| ------------------- | ---------------------------------------------------------------- |
| Status              | **Completed — Low Diversity (Normal Cell)**                      |
| Date                | 2026-02-25                                                       |
| Start time          | ~13:10                                                           |
| End time            | 18:27:25                                                         |
| Total search time   | **237.2 min (3.95 hours)**                                       |
| C_search            | 16                                                               |
| Batch size          | 16                                                               |
| Skip dropout (end)  | 0.5                                                              |
| Alpha warmup epochs | 15                                                               |
| Best val accuracy   | **Stage 3 → 89.45%**                                             |
| Root problem        | Stage 2→3 pruning left only `[none, skip_connect, dil_conv_3x3]` |

The search completed successfully (no collapse), but the final normal cell architecture is uniform: **all 8 operations are `dil_conv_3x3`**. Diversity in the reduce cell is partial (6× dil_conv_3x3 + 2× skip_connect). The genotype was nevertheless retrained and achieved **98.44% test accuracy** (best epoch 247/300, AUC=0.9999, EER=0.0066%).

---

## 2. Three-Stage Training Summary

| Stage   | Cells | Ops | Epochs | Final val_acc | Final train_acc | Duration |
| ------- | ----- | --- | ------ | ------------- | --------------- | -------- |
| Stage 1 | 5     | 8   | 50     | **89.09%**    | 99.73%          | ~93 min  |
| Stage 2 | 8     | 5   | 50     | **89.33%**    | 100.00%         | ~92 min  |
| Stage 3 | 11    | 3   | 50     | **89.45%**    | 99.85%          | ~52 min  |

Key observation: val_acc improved marginally across stages (+0.24%, +0.12%), showing healthy progressive learning, but train_acc reaching **100% by Stage 2** indicates the search model memorised training data despite regularisation.

---

## 3. Final Genotype

```
Normal: [('dil_conv_3x3', 1), ('dil_conv_3x3', 0),
         ('dil_conv_3x3', 1), ('dil_conv_3x3', 2),
         ('dil_conv_3x3', 2), ('dil_conv_3x3', 1),
         ('dil_conv_3x3', 4), ('dil_conv_3x3', 1)]

Reduce: [('dil_conv_3x3', 1), ('dil_conv_3x3', 0),
         ('skip_connect', 1), ('dil_conv_3x3', 2),
         ('dil_conv_3x3', 0), ('skip_connect', 1),
         ('dil_conv_3x3', 3), ('dil_conv_3x3', 1)]
```

**Normal cell decoded:**

```
node_2 = dil_conv_3x3(node_1) + dil_conv_3x3(node_0)
node_3 = dil_conv_3x3(node_1) + dil_conv_3x3(node_2)
node_4 = dil_conv_3x3(node_2) + dil_conv_3x3(node_1)
node_5 = dil_conv_3x3(node_4) + dil_conv_3x3(node_1)
Output = concat(node_2, node_3, node_4, node_5)
```

**Reduce cell decoded:**

```
node_2 = dil_conv_3x3(node_1) + dil_conv_3x3(node_0)
node_3 = skip_connect(node_1) + dil_conv_3x3(node_2)
node_4 = dil_conv_3x3(node_0) + skip_connect(node_1)
node_5 = dil_conv_3x3(node_3) + dil_conv_3x3(node_1)
Output = concat(node_2, node_3, node_4, node_5)
```

**Op diversity:**

- Normal cell: **8/8 = 100% dil_conv_3x3** ← no diversity
- Reduce cell: **6/8 dil_conv_3x3**, 2/8 skip_connect ← partial diversity

---

## 4. Pruning Chain — Step-by-Step

### Stage 1 → Stage 2 (8 ops → 5 ops)

```
Before: [none, skip_connect, sep_conv_3x3, sep_conv_5x5,
         dil_conv_3x3, dil_conv_5x5, avg_pool_3x3, max_pool_3x3]

Anti-collapse guard: forced dil_conv_3x3 (score=0.168) into kept set

After:  [none, skip_connect, sep_conv_5x5, dil_conv_3x3, dil_conv_5x5]

Eliminated: sep_conv_3x3, avg_pool_3x3, max_pool_3x3
```

**Analysis:** Healthy pruning. Three pooling/small-conv ops removed. Remaining ops include three distinct convolutional kernels (`sep_conv_5x5`, `dil_conv_3x3`, `dil_conv_5x5`). Stage 1 genotype itself showed diversity: `max_pool_3x3`, `sep_conv_5x5`, `dil_conv_5x5`, `dil_conv_3x3` all appeared.

---

### Stage 2 → Stage 3 (5 ops → 3 ops) ← **Root Cause**

```
Before: [none, skip_connect, sep_conv_5x5, dil_conv_3x3, dil_conv_5x5]

Anti-collapse guard: forced dil_conv_3x3 (score=0.256) into kept set

After:  [none, skip_connect, dil_conv_3x3]

Eliminated: sep_conv_5x5, dil_conv_5x5
```

**This is where diversity was permanently lost.**

After Stage 2→3 pruning, Stage 3 can only use 3 operations: `none`, `skip_connect`, `dil_conv_3x3`. With the old `STRUCTURAL_OPS = {'none', 'skip_connect'}` setting, **both** `none` and `skip_connect` were auto-kept as structural ops regardless of alpha score. This left only **1 slot** for a learnable conv operation, and that single slot was `dil_conv_3x3` (forced by anti-collapse guard). No other conv type could survive.

---

## 5. Stage 2 Alpha Distributions (Detailed)

Stage 2 ended with **diverse and healthy** alpha weights across all edges. The problem was **not** overly dominant alphas in Stage 2 — it was the pruning decision.

### Normal Cell Alphas — End of Stage 2

| Edge | none  | skip_connect | sep_conv_5x5 | dil_conv_3x3 | dil_conv_5x5 | Winner       |
| ---- | ----- | ------------ | ------------ | ------------ | ------------ | ------------ |
| 0→2  | 0.659 | 0.053        | 0.084        | **0.122**    | 0.081        | none (high!) |
| 1→2  | 0.088 | 0.256        | 0.101        | **0.279**    | 0.276        | dil_conv_3x3 |
| 0→3  | 0.346 | 0.081        | 0.121        | **0.273**    | 0.180        | none         |
| 1→3  | 0.053 | 0.155        | 0.079        | **0.490**    | 0.223        | dil_conv_3x3 |
| 2→3  | 0.061 | 0.120        | **0.502**    | 0.094        | 0.223        | sep_conv_5x5 |
| 0→4  | 0.146 | 0.058        | **0.514**    | 0.117        | 0.164        | sep_conv_5x5 |
| 1→4  | 0.073 | 0.092        | 0.217        | 0.175        | **0.443**    | dil_conv_5x5 |
| 2→4  | 0.050 | 0.046        | 0.133        | **0.428**    | 0.343        | dil_conv_3x3 |
| 3→4  | 0.058 | **0.306**    | 0.213        | 0.193        | 0.230        | skip_connect |
| 0→5  | 0.265 | 0.069        | 0.240        | 0.125        | **0.301**    | dil_conv_5x5 |
| 1→5  | 0.058 | 0.111        | 0.173        | **0.570**    | 0.087        | dil_conv_3x3 |
| 2→5  | 0.074 | 0.067        | 0.307        | **0.443**    | 0.109        | dil_conv_3x3 |
| 3→5  | 0.048 | 0.124        | **0.396**    | 0.256        | 0.176        | sep_conv_5x5 |
| 4→5  | 0.042 | 0.119        | 0.082        | **0.568**    | 0.189        | dil_conv_3x3 |

**Stage 2 Genotype (Normal):** `dil_conv_3x3`×4, `sep_conv_5x5`×2, `dil_conv_5x5`×1, `dil_conv_3x3`×1 — genuinely diverse.

### Cumulative Op Score (Normal Cell, averaged across all edges)

| Op               | Avg alpha  | Pruning fate                   |
| ---------------- | ---------- | ------------------------------ |
| none             | ~0.158     | Auto-kept (STRUCTURAL_OPS)     |
| skip_connect     | ~0.126     | Auto-kept (STRUCTURAL_OPS)     |
| **dil_conv_3x3** | **~0.270** | Kept (forced by anti-collapse) |
| sep_conv_5x5     | ~0.243     | ❌ **Eliminated**              |
| dil_conv_5x5     | ~0.215     | ❌ **Eliminated**              |

> **Critical insight:** `sep_conv_5x5` (avg 0.243) and `dil_conv_5x5` (avg 0.215) are both strong competitors, yet they were eliminated. This happened because the pruning needs exactly **3 ops total**, and STRUCTURAL_OPS already locked 2 of those 3 positions. Only 1 conv slot remained → highest-scoring conv (`dil_conv_3x3`, avg 0.270) won. `sep_conv_5x5` despite high Stage 2 alphas had no chance.

---

## 6. Stage 3 Alpha Distributions (Final)

After the forced reduction to `[none, skip_connect, dil_conv_3x3]`, Stage 3 shows expected alpha behaviour: `dil_conv_3x3` rapidly dominates (since it is the only learnable option for most edges).

### Normal Cell Alphas — End of Stage 3

| Edge | none  | skip_connect | dil_conv_3x3 | Winner                  |
| ---- | ----- | ------------ | ------------ | ----------------------- |
| 0→2  | 0.850 | 0.041        | **0.109**    | none (high — weak edge) |
| 1→2  | 0.126 | 0.394        | **0.480**    | dil_conv_3x3            |
| 0→3  | 0.797 | 0.071        | **0.132**    | none (weak input)       |
| 1→3  | 0.046 | 0.098        | **0.856**    | dil_conv_3x3            |
| 2→3  | 0.176 | 0.395        | **0.429**    | dil_conv_3x3            |
| 0→4  | 0.539 | 0.108        | **0.353**    | none                    |
| 1→4  | 0.188 | 0.253        | **0.560**    | dil_conv_3x3            |
| 2→4  | 0.045 | 0.053        | **0.902**    | dil_conv_3x3            |
| 3→4  | 0.118 | **0.480**    | 0.402        | skip_connect            |
| 0→5  | 0.662 | 0.099        | **0.238**    | none (weak)             |
| 1→5  | 0.069 | 0.141        | **0.790**    | dil_conv_3x3            |
| 2→5  | 0.117 | 0.099        | **0.784**    | dil_conv_3x3            |
| 3→5  | 0.107 | **0.363**    | 0.530        | dil_conv_3x3            |
| 4→5  | 0.043 | 0.111        | **0.845**    | dil_conv_3x3            |

**Observation:** `node_0` input edges consistently dominated by `none` (alpha ~0.66–0.85), indicating the network relies heavily on `node_1` (the previous cell's output) rather than `node_0` (the cell two steps back). `dil_conv_3x3` wins on 8 edges → genotype: 100% `dil_conv_3x3`.

### Normal Cell Reduce — Notable Pattern

Reduce cell shows `none` dominance in deep nodes (node_5 edges: none=0.49–0.53), contrasting with the Normal cell. The 2 `skip_connect` ops appear in node_3 and node_4 at `edge(1→3)` and `edge(1→4)`, providing some spatial identity mapping during downsampling.

---

## 7. Root Cause Analysis

### Primary Cause: STRUCTURAL_OPS = {'none', 'skip_connect'}

```python
# search.py (Run #3 configuration — OLD)
STRUCTURAL_OPS = {'none', 'skip_connect'}  # ← 2 auto-kept ops
MIN_CONV = 1

def prune_operations(primitives, alpha_normal, alpha_reduce, keep_k):
    ...
    # Auto-keep all structural ops regardless of alpha
    for name in STRUCTURAL_OPS:
        if name in primitive_set:
            kept.add(name)
    ...
```

**The math of the problem:**

| Stage       | keep_k | Structural auto-kept   | Conv slots free                        |
| ----------- | ------ | ---------------------- | -------------------------------------- |
| Stage 1 → 2 | 5      | 2 (none, skip_connect) | **3 conv slots** → diverse (OK)        |
| Stage 2 → 3 | 3      | 2 (none, skip_connect) | **1 conv slot** → only 1 conv survives |

In Stage 2→3, with keep_k=3 and 2 auto-kept structural ops, only **1** conv op could be selected by the anti-collapse guard. Since `dil_conv_3x3` outscored `sep_conv_5x5` and `dil_conv_5x5`, it won the single conv slot. Both other conv variants (sep_conv_5x5 avg=0.243, dil_conv_5x5 avg=0.215) were discarded despite being strong candidates.

### Secondary Cause: MIN_CONV = 1

With `MIN_CONV=1`, the pruning was satisfied as long as **any single conv** was in the final set. This allowed the degenerate case of 1 conv + 2 structural ops to pass the guard.

### Why anti-collapse guard didn't help

The anti-collapse guard forced inclusion of `dil_conv_3x3` — which was correct (prevented a collapse to `none`+`skip_connect`). However, it only forces **one** specific conv, not **N** diverse convs. The guard was designed to prevent full collapse (like Run #1 where max_pool dominated), not to enforce diversity among conv types.

### Interaction with 'none' auto-keep

`none` being in STRUCTURAL_OPS means ~50% of Stage 3 alphas had `none` as a competitor taking probability mass. Many `node_0` input edges had `none` dominate (alpha 0.53–0.85), suggesting the 11-cell Stage 3 model found the distant (`node_0`) input less useful. This is an indirect signal that STRUCTURAL_OPS locking `none` wasted valuable capacity.

---

## 8. Comparison to Run #1 Collapse

| Dimension          | Run #1 (Collapse)                                               | Run #3 (Low Diversity)                                       |
| ------------------ | --------------------------------------------------------------- | ------------------------------------------------------------ |
| Root cause         | C_search=8 too small, all conv ops underfit → pooling dominated | STRUCTURAL_OPS locked 2/3 final slots → 1 conv slot only     |
| Failure mode       | max_pool_3x3 alpha=0.868 in Stage 3                             | All 8 normal cell ops = dil_conv_3x3                         |
| Stage 3 primitives | [none, skip_connect, max_pool_3x3]                              | [none, skip_connect, dil_conv_3x3]                           |
| Val accuracy       | Collapsed before completion (~poor)                             | Healthy: 89.45%                                              |
| Retrain outcome    | Not commissioned (collapsed genotype)                           | **98.44%** (successfully retrained)                          |
| Fix applied        | C_search: 8→16                                                  | STRUCTURAL_OPS={'skip_connect'} (removed 'none'), MIN_CONV=2 |

---

## 9. Fix Applied (Before Run #4)

```python
# search.py (UPDATED — Run #4 config)
STRUCTURAL_OPS = {'skip_connect'}  # ← Only 1 auto-kept op (removed 'none')
MIN_CONV = 2                        # ← Require at least 2 conv ops in final set

def prune_operations(primitives, alpha_normal, alpha_reduce, keep_k):
    ...
    # With keep_k=3 and STRUCTURAL_OPS=1:
    # 1 auto-kept + 2 conv slots → 2 diverse convs survive
```

**Expected Stage 2→3 pruning with fix:**

| Stage       | keep_k | Structural auto-kept  | Conv slots free                         |
| ----------- | ------ | --------------------- | --------------------------------------- |
| Stage 2 → 3 | 3      | 1 (skip_connect only) | **2 conv slots** → 2 diverse conv types |

With keep_k=3 and only 1 structural auto-kept, 2 conv ops can survive. MIN_CONV=2 enforces this minimum regardless. `none` must now compete fairly with conv ops on alpha score, and high-scoring ops like `sep_conv_5x5` (avg=0.243) and `dil_conv_3x3` (avg=0.270) would both survive.

---

## 10. Retrain Results (Run #3 Genotype)

Despite low architectural diversity, the Run #3 genotype was successfully retrained:

| Metric            | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| Architecture      | Normal=100% dil_conv_3x3 / Reduce=75% dil_conv_3x3 + 25% skip_connect |
| C_init            | 8                                                                     |
| Num cells         | 8                                                                     |
| Epochs            | 300                                                                   |
| Best epoch        | 247                                                                   |
| **Test accuracy** | **98.44%**                                                            |
| AUC               | 0.9999                                                                |
| EER               | 0.0066%                                                               |
| Parameters        | 301,980 (~302K)                                                       |
| Model size        | 1.17 MB                                                               |
| GPU latency       | 10.1 ms                                                               |

**Interpretation:** Even a degenerate (low-diversity) architecture achieves high accuracy on palm vein recognition when properly trained. The homogeneous `dil_conv_3x3` stack provides consistent receptive field coverage (dilated 3×3 = effective 5×5 field) and benefits from the dataset's limited variability (834 classes × 8 samples). However, compared to a **diverse architecture** (expected from Run #4), the performance ceiling may be lower.

**KD Gap:** Teacher models (ResNet50, EfficientNet-V2M, ConvNeXt-B) achieve 100% accuracy. The 1.56% gap from teacher to this student is small, which limits the marginal benefit of KD with this exact model.

---

## 11. File Index

```
search_run3_low_diversity/
├── search.log                  ← Full 5980-line training log
├── search_log.csv              ← Per-epoch metrics CSV
├── search_summary.json         ← Final config + genotype summary
├── genotype_final.json         ← Genotype in JSON format
├── genotype_final.txt          ← Genotype in human-readable format
├── genotype_final.png          ← Architecture visualisation
├── stage_1/                    ← Stage 1 intermediate artefacts
├── stage_2/                    ← Stage 2 intermediate artefacts
├── stage_3/                    ← Stage 3 intermediate artefacts
├── stage_1_checkpoint.pt       ← Stage 1 model weights (~6.2 MB)
├── stage_2_checkpoint.pt       ← Stage 2 model weights (~6.2 MB)
├── stage_3_checkpoint.pt       ← Stage 3 model weights (~2.9 MB)
└── RUN3_LOW_DIVERSITY_ANALYSIS.md   ← This file
```

---

## 12. Timeline

```
13:10   Search started (approx.)
16:00   Stage 1 complete — val_acc=89.09%
16:00   Stage 1→2 pruning: 8 ops → 5 ops
        [none, skip_connect, sep_conv_5x5, dil_conv_3x3, dil_conv_5x5]
16:00   Stage 2 begins
17:33   Stage 2 complete — val_acc=89.33%
17:33   Stage 2→3 pruning: 5 ops → 3 ops ← DIVERSITY LOST HERE
        [none, skip_connect, dil_conv_3x3]
17:33   Stage 3 begins
18:27   Stage 3 complete — val_acc=89.45%
18:27   Final genotype saved
18:27   Search complete — total_time=237.2 min
```

---

_Generated post-hoc from search.log and search_summary.json — 2026-02-25_

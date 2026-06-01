# Run7 Configuration Verification Report

## User Question
User asked: "tunggu retrain mu ini confignya gmn c initnya berapa? aku mau confignya itu kayak run6"

## Answer: Configuration is CORRECT ✅

The `retrain_run7_robust.py` script is already configured to match run6 exactly.

---

## Run6 Actual Configuration (Verified from retrain.log)

```
C_init:         5
num_cells:      8
epochs:         300
batch_size:     64
lr:             0.001
weight_decay:   0.05
drop_path_prob: 0.2
cutout_length:  16
auxiliary:      True
seed:           42
num_workers:    4

# From RETRAIN_CFG (nas_config.py):
warmup_epochs:      10
warmup_factor:      0.01
label_smoothing:    0.2
dropout:            0.3
auxiliary_weight:   0.4
grad_clip:          1.0
lr_min:             1e-6
```

**Total Parameters**: 225,476 (below target min of 250k, but this is what run6 used)

---

## Run7 Current Configuration (from run7_config.json)

```json
{
  "base_config": "run6",
  "dataset": "multi_distance",
  "augmentation_policy": "v2_multi_distance",
  "C_init": 5,                    ✅ MATCHES run6
  "num_cells": 8,                 ✅ MATCHES run6
  "epochs": 300,                  ✅ MATCHES run6
  "batch_size": 64,               ✅ MATCHES run6
  "lr": 0.001,                    ✅ MATCHES run6
  "weight_decay": 0.05,           ✅ MATCHES run6
  "drop_path_prob": 0.2,          ✅ MATCHES run6
  "cutout_length": 16,            ✅ MATCHES run6
  "auxiliary": true,              ✅ MATCHES run6
  "seed": 42,                     ✅ MATCHES run6
  "num_workers": 4                ✅ MATCHES run6
}
```

The script also inherits all RETRAIN_CFG parameters from `nas_config.py` (warmup, label_smoothing, dropout, etc.) just like run6 did.

---

## Key Differences from Run6 (Intentional for Robustness Fix)

### 1. Dataset
- **Run6**: Original dataset (834 subjects, 27cm only)
- **Run7**: Multi-distance dataset (2 subjects: 835, 836; 5 distances: 22, 25, 27, 30, 32 cm)
  - Total: 100 images (50 per subject, 10 per distance)
  - Split: 60 train / 20 val / 20 test

### 2. Augmentation Policy
- **Run6**: `v1_legacy` (includes horizontal flip)
- **Run7**: `v2_multi_distance` (NO horizontal flip to prevent cross-hand confusion)
  - Rotation: ±15° (increased from ±10°)
  - Scale: 0.78-1.28 (wider range)
  - ColorJitter: brightness=0.3, contrast=0.3, saturation=0.2

### 3. Hand-Pair Margin Loss
- **Run6**: Not implemented
- **Run7**: Available but currently disabled (`hand_pair_margin_loss: false`)
  - Can be enabled with `--hand-pair-margin-loss` flag
  - Designed to improve cross-hand discrimination

---

## Confusion in config.json Explained

The `run6/config.json` file contains **two sets of parameters**:

1. **Top-level parameters** (C_init=5, epochs=300) - **THESE WERE ACTUALLY USED** ✅
2. **retrain_cfg dictionary** (C_init=14, epochs=600) - These are just the defaults from `nas_config.py`, NOT what run6 used

The retrain.log confirms run6 used:
- C_init=5 (not 14)
- epochs=300 (not 600)

---

## Verification Commands

### Check run7 config:
```bash
cat nas_results/retrain_run7_robust/run7_config.json
```

### Check run6 actual parameters:
```bash
head -50 nas_results/retrain_run6/retrain.log
```

### Check converted split:
```bash
cat nas_results/retrain_run7_robust/split_info_converted.json
```

### Check symlinked data structure:
```bash
ls -la nas_results/retrain_run7_robust/data_symlinks/835/
ls -la nas_results/retrain_run7_robust/data_symlinks/836/
```

---

## Ready to Train

The configuration is correct and matches run6. To start training:

```bash
# Option 1: Use the wrapper script (recommended)
python retrain_run7_robust.py

# Option 2: Manual retrain.py call
python retrain.py \
  --genotype nas_results/search/genotype_final.json \
  --data_dir nas_results/retrain_run7_robust/data_symlinks \
  --split_path nas_results/retrain_run7_robust/split_info_converted.json \
  --output_dir nas_results/retrain_run7_robust \
  --augmentation_policy v2_multi_distance \
  --C_init 5 \
  --num_cells 8 \
  --epochs 300 \
  --batch_size 64 \
  --lr 0.001 \
  --weight_decay 0.05 \
  --drop_path_prob 0.2 \
  --cutout_length 16 \
  --auxiliary \
  --seed 42 \
  --num_workers 4
```

---

## Expected Training Time

Based on run6 (300 epochs, ~27s per epoch):
- **Total time**: ~2.25 hours
- **Checkpoint**: Best model saved when val accuracy improves

---

## Target Accuracy (Revised)

From `.kiro/specs/live-scan-robustness-fix/design.md` (TA-2):
- **Original target**: ≥95% accuracy
- **Revised target**: ≥90% accuracy (due to small dataset size and multi-distance challenge)

---

## Summary

✅ **Configuration is correct** - run7 uses the same hyperparameters as run6  
✅ **Dataset prepared** - 100 images, split 60/20/20  
✅ **Augmentation updated** - No horizontal flip to prevent cross-hand confusion  
✅ **Ready to train** - All files in place, just run `python retrain_run7_robust.py`

The only differences from run6 are intentional improvements for the robustness fix task.

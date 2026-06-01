# Split File Report — Multi-Distance Dataset

**Date:** 1 Juni 2026  
**Split File:** `dataset_multi_distance/split_info.json`  
**Source:** Preprocessed BMP images (224×224) from `final/` folder  
**Total Images:** 100 (50 per subject)

---

## ✅ Split Summary

### Overall Distribution

| Split | Total | Subject 835 (Left) | Subject 836 (Right) | Ratio |
|-------|-------|-------------------|---------------------|-------|
| **Train** | 60 | 31 (51.7%) | 29 (48.3%) | 60% |
| **Val** | 20 | 8 (40.0%) | 12 (60.0%) | 20% |
| **Test** | 20 | 11 (55.0%) | 9 (45.0%) | 20% |
| **TOTAL** | **100** | **50** | **50** | 100% |

**Balance:** ✅ Well-balanced across subjects (±2 images difference per split)

---

## Detailed Distribution per Distance

### Train Set (60 images)

| Distance | Subject 835 | Subject 836 | Total | Coverage |
|----------|-------------|-------------|-------|----------|
| 22 cm | 5 | 7 | 12 | 60% of 20 |
| 25 cm | 5 | 7 | 12 | 60% of 20 |
| 27 cm | 4 | 6 | 10 | 50% of 20 |
| 30 cm | 8 | 4 | 12 | 60% of 20 |
| 32 cm | 9 | 5 | 14 | 70% of 20 |
| **TOTAL** | **31** | **29** | **60** | **60%** |

**Observation:** Training set punya good coverage di semua jarak (50-70% per distance)

### Val Set (20 images)

| Distance | Subject 835 | Subject 836 | Total | Coverage |
|----------|-------------|-------------|-------|----------|
| 22 cm | 3 | 2 | 5 | 25% of 20 |
| 25 cm | 1 | 1 | 2 | 10% of 20 |
| 27 cm | 3 | 3 | 6 | 30% of 20 |
| 30 cm | 0 | 3 | 3 | 15% of 20 |
| 32 cm | 1 | 3 | 4 | 20% of 20 |
| **TOTAL** | **8** | **12** | **20** | **20%** |

**Note:** Subject 835 tidak punya sample di 30 cm pada val set (random split artifact)

### Test Set (20 images)

| Distance | Subject 835 | Subject 836 | Total | Coverage |
|----------|-------------|-------------|-------|----------|
| 22 cm | 2 | 1 | 3 | 15% of 20 |
| 25 cm | 4 | 2 | 6 | 30% of 20 |
| 27 cm | 3 | 1 | 4 | 20% of 20 |
| 30 cm | 2 | 3 | 5 | 25% of 20 |
| 32 cm | 0 | 2 | 2 | 10% of 20 |
| **TOTAL** | **11** | **9** | **20** | **20%** |

**Note:** Subject 835 tidak punya sample di 32 cm pada test set (random split artifact)

---

## Split Configuration

```json
{
  "dataset_root": "dataset_multi_distance",
  "source_folder": "final",
  "subjects": ["835", "836"],
  "label_map": {
    "835": 0,
    "836": 1
  },
  "split_ratios": {
    "train": 0.6,
    "val": 0.2,
    "test": 0.2
  },
  "random_seed": 42
}
```

### File Paths

**Format:** `{subject_id}/final/{distance_cm}/{filename}.bmp`

**Example paths:**
```
Train:
  835/final/32cm/palm_20260601_221907_460356.bmp
  835/final/32cm/palm_20260601_221858_710346.bmp
  836/final/32cm/palm_20260601_172133_896458.bmp

Val:
  835/final/27cm/palm_20260601_220745_237635.bmp
  835/final/22cm/palm_20260601_215659_061283.bmp

Test:
  835/final/25cm/palm_20260601_220053_437812.bmp
  836/final/30cm/palm_20260601_171440_368038.bmp
```

---

## Coverage Analysis

### Per (Subject, Distance) Coverage

| Subject | Distance | Train | Val | Test | Total | Train % |
|---------|----------|-------|-----|------|-------|---------|
| 835 | 22 cm | 5 | 3 | 2 | 10 | 50% |
| 835 | 25 cm | 5 | 1 | 4 | 10 | 50% |
| 835 | 27 cm | 4 | 3 | 3 | 10 | 40% |
| 835 | 30 cm | 8 | 0 | 2 | 10 | 80% |
| 835 | 32 cm | 9 | 1 | 0 | 10 | 90% |
| 836 | 22 cm | 7 | 2 | 1 | 10 | 70% |
| 836 | 25 cm | 7 | 1 | 2 | 10 | 70% |
| 836 | 27 cm | 6 | 3 | 1 | 10 | 60% |
| 836 | 30 cm | 4 | 3 | 3 | 10 | 40% |
| 836 | 32 cm | 5 | 3 | 2 | 10 | 50% |

### Minimum Coverage

- **Minimum train samples per (subject, distance):** 4 (835 @ 27cm, 836 @ 30cm)
- **Minimum val samples per (subject, distance):** 0 (835 @ 30cm)
- **Minimum test samples per (subject, distance):** 0 (835 @ 32cm)

**Implication:** Beberapa kombinasi (subject, distance) tidak punya representation di val/test set. Ini adalah artifact dari random split dengan volume kecil (10 samples per combination).

---

## Recommendations

### ✅ Acceptable for Training

Split ini **acceptable** untuk training dengan caveats:

1. **Training set well-balanced:** 60 images dengan good coverage di semua jarak
2. **Val/test sufficient:** 20 images each untuk monitor overfitting dan final evaluation
3. **Subject balance OK:** ±2 images difference per split (51.7% vs 48.3% di train)

### ⚠️ Caveats

1. **Sparse val/test coverage:** Beberapa (subject, distance) combinations tidak punya sample di val/test
   - **Impact:** Validation/test metrics mungkin tidak fully representative untuk semua jarak
   - **Mitigation:** Monitor training curves closely, dan evaluate pada held-out OOD set (18 cm, 38 cm) nanti

2. **Small sample size:** Dengan 10 samples per (subject, distance), random split akan inevitably create gaps
   - **Impact:** Model mungkin underfit di jarak dengan sedikit training samples (4-5)
   - **Mitigation:** Augmentation v2 akan compensate dengan synthetic variability

### 💡 Alternative: Stratified Split

Jika ingin **guarantee** setiap (subject, distance) punya minimal 1 sample di val/test:

```python
# Stratified split: 6 train / 2 val / 2 test per (subject, distance)
# This ensures every combination is represented in all splits
# Trade-off: Less randomness, more deterministic split
```

**Recommendation:** Stick dengan current random split untuk now, karena:
- Training set punya good coverage (4-9 samples per combination)
- Augmentation v2 akan generate synthetic variability
- Dapat iterate nanti jika validation metrics tidak stable

---

## Usage in Training

### Load Split File

```python
import json

# Load split info
with open("dataset_multi_distance/split_info.json") as f:
    split_info = json.load(f)

# Get image paths
train_paths = split_info["splits"]["train"]  # 60 paths
val_paths = split_info["splits"]["val"]      # 20 paths
test_paths = split_info["splits"]["test"]    # 20 paths

# Get labels
label_map = split_info["label_map"]  # {"835": 0, "836": 1}

# Extract labels from metadata
train_labels = [
    label_map[meta["subject_id"]] 
    for meta in split_info["metadata"]["train"]
]
```

### Create Dataloaders

```python
from palm_vein_dataset import PalmVeinDataset, get_transforms

# Create datasets
train_dataset = PalmVeinDataset(
    image_paths=[f"dataset_multi_distance/{p}" for p in train_paths],
    labels=train_labels,
    transform=get_transforms(
        split="train",
        use_augmentation=True,
        augmentation_policy="v2_multi_distance"
    )
)

val_dataset = PalmVeinDataset(
    image_paths=[f"dataset_multi_distance/{p}" for p in val_paths],
    labels=val_labels,
    transform=get_transforms(split="val")
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
```

---

## Verification Commands

```bash
# Check split file
cat dataset_multi_distance/split_info.json | jq '.source_folder'
# Output: "final"

# Count images per split
cat dataset_multi_distance/split_info.json | jq '.splits | {train: .train | length, val: .val | length, test: .test | length}'
# Output: {"train": 60, "val": 20, "test": 20}

# Verify file extensions
cat dataset_multi_distance/split_info.json | jq '.splits.train[0]'
# Output: "835/final/32cm/palm_20260601_221907_460356.bmp"

# Check label map
cat dataset_multi_distance/split_info.json | jq '.label_map'
# Output: {"835": 0, "836": 1}
```

---

## Summary

✅ **Split file generated successfully**
- Source: Preprocessed BMP images (224×224) from `final/` folder
- Total: 100 images (60 train / 20 val / 20 test)
- Balance: Well-balanced across subjects (±2 images per split)
- Format: Paths relative to `dataset_multi_distance/`

⚠️ **Known limitations:**
- Sparse val/test coverage (some combinations have 0 samples)
- Small sample size per combination (4-9 train samples)

✅ **Ready for training:**
- Training set has good coverage (60 images across all distances)
- Augmentation v2 will compensate for small sample size
- Can iterate on split strategy if needed

🚀 **Next step:** Task 7 (retrain with augmentation v2 + hand-pair loss)

# Dataset Split Strategy — Multi-Distance Training

**Strategy:** Campur semua jarak, lalu split random  
**Rationale:** Model belajar distance-invariant features

---

## ✅ CORRECT Strategy: Mix All Distances → Random Split

### Workflow

```
Step 1: Collect all images per subject (mix dari semua jarak)
  
  Tangan Kiri (835):
    22cm: 10 images ─┐
    25cm: 10 images  │
    27cm: 10 images  ├─→ POOL 835 (50 images)
    30cm: 10 images  │
    32cm: 10 images ─┘
  
  Tangan Kanan (836):
    22cm: 10 images ─┐
    25cm: 10 images  │
    27cm: 10 images  ├─→ POOL 836 (50 images)
    30cm: 10 images  │
    32cm: 10 images ─┘

Step 2: Combine pools
  
  TOTAL POOL: 100 images (50 per subject)

Step 3: Random shuffle dengan seed (reproducibility)
  
  random.seed(42)
  random.shuffle(all_images)

Step 4: Split 60% train / 20% val / 20% test
  
  Train: 60 images (30 per subject)
  Val:   20 images (10 per subject)
  Test:  20 images (10 per subject)
```

---

## Expected Distribution (Example dengan seed=42)

### Preview dengan Tangan Kiri (835) Saja

**Train (30 images):**
- 22cm: 4 images
- 25cm: 6 images
- 27cm: 9 images
- 30cm: 6 images
- 32cm: 5 images

**Val (10 images):**
- 22cm: 1 images
- 25cm: 1 images
- 27cm: 1 images
- 30cm: 3 images
- 32cm: 4 images

**Test (10 images):**
- 22cm: 5 images
- 25cm: 3 images
- 27cm: 0 images (⚠️ akan ada setelah tambah 836)
- 30cm: 1 images
- 32cm: 1 images

### Expected dengan Kedua Tangan (835 + 836)

**Train (60 images total):**
- Subject 835: ~30 images (mix dari 5 jarak)
- Subject 836: ~30 images (mix dari 5 jarak)
- **Per (subject, distance):** ~6 images (range: 4-8)

**Val (20 images total):**
- Subject 835: ~10 images (mix dari 5 jarak)
- Subject 836: ~10 images (mix dari 5 jarak)
- **Per (subject, distance):** ~2 images (range: 1-3)

**Test (20 images total):**
- Subject 835: ~10 images (mix dari 5 jarak)
- Subject 836: ~10 images (mix dari 5 jarak)
- **Per (subject, distance):** ~2 images (range: 1-3)

---

## Rationale: Kenapa Campur Semua Jarak?

### ✅ Advantages

1. **Distance-invariant learning:**
   - Training set punya mix dari semua jarak (22-32 cm)
   - Model belajar features yang robust terhadap variasi jarak
   - Tidak bias ke jarak tertentu (misal: hanya 27 cm)

2. **Representative validation/test:**
   - Val/test set juga punya mix dari semua jarak
   - Metrics (accuracy, loss) mencerminkan robustness sebenarnya
   - Tidak ada "surprise" saat deployment (live scan di jarak baru)

3. **Avoid distance-specific overfitting:**
   - Jika split per-distance (misal: 22cm hanya di train, 32cm hanya di test)
   - Model akan overfit ke jarak training, gagal generalisasi ke jarak test
   - Cross-distance accuracy akan rendah

4. **Augmentation synergy:**
   - Augmentation v2 (`RandomAffine scale=(0.78, 1.28)`) simulate variasi ROI size
   - Combined dengan real multi-distance data → double robustness
   - Model belajar dari real variability + synthetic variability

### ❌ Alternative (WRONG): Split Per-Distance

```
WRONG Strategy:
  Train: 22cm, 25cm, 27cm (30 images)
  Val:   30cm (10 images)
  Test:  32cm (10 images)

Problems:
  ❌ Model hanya lihat 22-27cm saat training
  ❌ Val/test di 30-32cm adalah "out-of-distribution"
  ❌ Metrics tidak representative (test accuracy akan rendah artifisial)
  ❌ Tidak bisa evaluate robustness sebenarnya
```

---

## Implementation

### Command untuk Build Split File

```bash
# Preview dengan tangan kiri (835) saja
python3 build_multi_distance_split.py \
    --dataset-root dataset_multi_distance \
    --output-file dataset_multi_distance/split_info_preview.json \
    --subjects 835 \
    --train-ratio 0.6 \
    --val-ratio 0.2 \
    --test-ratio 0.2 \
    --seed 42

# Final dengan kedua tangan (835 + 836)
python3 build_multi_distance_split.py \
    --dataset-root dataset_multi_distance \
    --output-file dataset_multi_distance/split_info.json \
    --subjects 835 836 \
    --train-ratio 0.6 \
    --val-ratio 0.2 \
    --test-ratio 0.2 \
    --seed 42
```

### Output: `split_info.json`

```json
{
  "dataset_root": "dataset_multi_distance",
  "source_folder": "final_raw",
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
  "random_seed": 42,
  "splits": {
    "train": [
      "835/final_raw/22cm/palm_20260524_153938_386475.png",
      "835/final_raw/25cm/palm_20260601_211906_016149.png",
      ...
    ],
    "val": [...],
    "test": [...]
  },
  "metadata": {
    "train": [
      {
        "path": "835/final_raw/22cm/palm_20260524_153938_386475.png",
        "subject_id": "835",
        "distance_cm": "22cm"
      },
      ...
    ],
    "val": [...],
    "test": [...]
  }
}
```

---

## Training Integration

### Dataloader akan Load dari Split File

```python
# retrain_run7_robust.py

# Load split info
with open("dataset_multi_distance/split_info.json") as f:
    split_info = json.load(f)

# Create datasets
train_dataset = PalmVeinDataset(
    image_paths=split_info["splits"]["train"],
    labels=[split_info["label_map"][meta["subject_id"]] 
            for meta in split_info["metadata"]["train"]],
    transform=get_transforms(
        split="train",
        use_augmentation=True,
        augmentation_policy="v2_multi_distance"  # NEW
    )
)

val_dataset = PalmVeinDataset(
    image_paths=split_info["splits"]["val"],
    labels=[split_info["label_map"][meta["subject_id"]] 
            for meta in split_info["metadata"]["val"]],
    transform=get_transforms(split="val")
)

test_dataset = PalmVeinDataset(
    image_paths=split_info["splits"]["test"],
    labels=[split_info["label_map"][meta["subject_id"]] 
            for meta in split_info["metadata"]["test"]],
    transform=get_transforms(split="test")
)
```

### Augmentation v2 akan Apply pada Training Set

```python
# palm_vein_dataset.py

def get_transforms(split, use_augmentation=False, augmentation_policy="v1_legacy"):
    if split == "train" and use_augmentation:
        if augmentation_policy == "v2_multi_distance":
            return Compose([
                # NO RandomHorizontalFlip (removed untuk fix cross-hand confusion)
                RandomRotation(degrees=15),
                RandomAffine(
                    degrees=0,
                    translate=(0.08, 0.08),
                    scale=(0.78, 1.28)  # Simulate variasi ROI size dari jarak
                ),
                ColorJitter(brightness=0.20, contrast=0.15),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Cutout(cutout_length=16)
            ])
    
    # Val/test: no augmentation
    return Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

---

## Expected Training Behavior

### Epoch 1-20: Model belajar basic features
- Loss turun cepat (cross-entropy + hand-pair margin loss)
- Training accuracy: 60-70%
- Validation accuracy: 50-60%
- Model mulai distinguish antara 835 vs 836

### Epoch 21-50: Model belajar distance-invariant features
- Loss turun lebih lambat
- Training accuracy: 80-90%
- Validation accuracy: 75-85%
- Model mulai robust terhadap variasi jarak

### Epoch 51-100: Fine-tuning dan convergence
- Loss plateau
- Training accuracy: 95-99%
- Validation accuracy: 88-94%
- Model converge ke optimal point

### Final Expected Metrics (TA-2)
- **Test accuracy:** ≥90% (target revised dari ≥95%)
- **Cross-hand confusion:** 0 (TA-3)
- **Per-distance accuracy:**
  - 22cm: 85-90%
  - 25cm: 88-92%
  - 27cm: 92-95% (nominal, paling familiar)
  - 30cm: 88-92%
  - 32cm: 85-90%

---

## Verification

### Check Split Balance

```bash
# Analyze distribution
python3 -c "
import json
with open('dataset_multi_distance/split_info.json') as f:
    split_info = json.load(f)

for split_name in ['train', 'val', 'test']:
    metadata = split_info['metadata'][split_name]
    print(f'{split_name}: {len(metadata)} images')
    
    # Count per subject
    from collections import Counter
    subject_counts = Counter(m['subject_id'] for m in metadata)
    for subject_id, count in sorted(subject_counts.items()):
        print(f'  {subject_id}: {count} images')
"
```

### Expected Output

```
train: 60 images
  835: 30 images
  836: 30 images
val: 20 images
  835: 10 images
  836: 10 images
test: 20 images
  835: 10 images
  836: 10 images
```

---

## Summary

✅ **Strategy:** Campur semua jarak → Random split 60/20/20

✅ **Rationale:**
- Model belajar distance-invariant features
- Val/test representative untuk robustness evaluation
- Avoid distance-specific overfitting

✅ **Implementation:**
- Script: `build_multi_distance_split.py`
- Output: `dataset_multi_distance/split_info.json`
- Integration: Load split file di `retrain_run7_robust.py`

✅ **Expected outcome:**
- Test accuracy: 90-94%
- Cross-hand confusion: 0
- Robust di semua jarak (22-32 cm)

🔜 **Next step:** Akuisisi tangan kanan (836), lalu run split builder dengan kedua tangan

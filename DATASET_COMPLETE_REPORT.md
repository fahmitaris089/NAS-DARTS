# Dataset Complete Report — Multi-Distance Palm Vein

**Date:** 1 Juni 2026  
**Status:** ✅ COMPLETE — Kedua tangan ready untuk training  
**Total Images:** 100 (50 per tangan, 10 per jarak)

---

## ✅ Dataset Summary

### Subject 835 — Tangan KIRI (Left Hand)

| Jarak | Raw (final_raw) | Preprocessed (final) | Quality (Lap Var) | Status |
|-------|-----------------|----------------------|-------------------|--------|
| 22 cm | 10 PNG | 10 BMP + 10 JSON | 19.9 ± 3.8 | ✅ |
| 25 cm | 10 PNG | 10 BMP + 10 JSON | 16.6 ± 2.0 | ✅ |
| 27 cm | 10 PNG | 10 BMP + 10 JSON | 15.7 ± 2.8 | ✅ |
| 30 cm | 10 PNG | 10 BMP + 10 JSON | 13.3 ± 3.2 | ✅ |
| 32 cm | 10 PNG | 10 BMP + 10 JSON | 12.4 ± 2.0 | ✅ |
| **TOTAL** | **50 PNG** | **50 BMP + 50 JSON** | **15.6 ± 3.4** | ✅ |

### Subject 836 — Tangan KANAN (Right Hand)

| Jarak | Raw (final_raw) | Preprocessed (final) | Quality (Lap Var) | Status |
|-------|-----------------|----------------------|-------------------|--------|
| 22 cm | 10 PNG | 10 BMP + 10 JSON | 20.8 ± 2.1 | ✅ |
| 25 cm | 10 PNG | 10 BMP + 10 JSON | 20.0 ± 3.1 | ✅ |
| 27 cm | 10 PNG | 10 BMP + 10 JSON | 16.0 ± 3.1 | ✅ |
| 30 cm | 10 PNG | 10 BMP + 10 JSON | 13.6 ± 1.1 | ✅ |
| 32 cm | 10 PNG | 10 BMP + 10 JSON | 11.1 ± 1.5 | ✅ |
| **TOTAL** | **50 PNG** | **50 BMP + 50 JSON** | **16.3 ± 4.2** | ✅ |

### Combined Dataset

| Metric | Value |
|--------|-------|
| **Total subjects** | 2 (835 = left, 836 = right) |
| **Total distances** | 5 (22, 25, 27, 30, 32 cm) |
| **Total raw images** | 100 PNG |
| **Total preprocessed** | 100 BMP + 100 JSON |
| **Images per subject** | 50 |
| **Images per distance** | 20 (10 per subject) |
| **Average quality** | 15.9 ± 3.8 (laplacian variance) |

---

## Dataset Structure

```
dataset_multi_distance/
├── 835/                        # ✅ Tangan KIRI (Left Hand)
│   ├── final_raw/              # 50 raw PNG (selected best)
│   │   ├── 22cm/ (10)
│   │   ├── 25cm/ (10)
│   │   ├── 27cm/ (10)
│   │   ├── 30cm/ (10)
│   │   ├── 32cm/ (10)
│   │   └── selection_summary.json
│   ├── final/                  # 50 preprocessed BMP + JSON
│   │   ├── 22cm/ (10 BMP + 10 JSON)
│   │   ├── 25cm/ (10 BMP + 10 JSON)
│   │   ├── 27cm/ (10 BMP + 10 JSON)
│   │   ├── 30cm/ (10 BMP + 10 JSON)
│   │   └── 32cm/ (10 BMP + 10 JSON)
│   ├── 22cm/                   # Original captures (all)
│   │   ├── raw/
│   │   ├── processed/
│   │   ├── final/
│   │   └── visualizations/
│   └── ... (25cm, 27cm, 30cm, 32cm)
│
└── 836/                        # ✅ Tangan KANAN (Right Hand)
    ├── final_raw/              # 50 raw PNG (selected best)
    │   ├── 22cm/ (10)
    │   ├── 25cm/ (10)
    │   ├── 27cm/ (10)
    │   ├── 30cm/ (10)
    │   ├── 32cm/ (10)
    │   └── selection_summary.json
    ├── final/                  # 50 preprocessed BMP + JSON
    │   ├── 22cm/ (10 BMP + 10 JSON)
    │   ├── 25cm/ (10 BMP + 10 JSON)
    │   ├── 27cm/ (10 BMP + 10 JSON)
    │   ├── 30cm/ (10 BMP + 10 JSON)
    │   └── 32cm/ (10 BMP + 10 JSON)
    ├── 22cm/                   # Original captures (all)
    │   ├── raw/
    │   ├── processed/
    │   ├── final/
    │   └── visualizations/
    └── ... (25cm, 27cm, 30cm, 32cm)
```

---

## Quality Analysis

### Quality Gradient (Expected Pattern)

Both hands show expected quality gradient:
- **22 cm:** Highest quality (~20), jarak dekat = intensitas NIR tinggi
- **27 cm:** Medium quality (~16), jarak nominal
- **32 cm:** Lower quality (~11-12), jarak jauh = intensitas NIR rendah

**Observation:** Tangan kanan (836) memiliki quality slightly lebih tinggi di jarak dekat (22-25 cm), tapi overall comparable dengan tangan kiri (835).

### Quality Comparison

| Jarak | Left (835) | Right (836) | Difference |
|-------|------------|-------------|------------|
| 22 cm | 19.9 | 20.8 | +0.9 (right better) |
| 25 cm | 16.6 | 20.0 | +3.4 (right better) |
| 27 cm | 15.7 | 16.0 | +0.3 (comparable) |
| 30 cm | 13.3 | 13.6 | +0.3 (comparable) |
| 32 cm | 12.4 | 11.1 | -1.3 (left better) |
| **Avg** | **15.6** | **16.3** | **+0.7** |

**Interpretation:** Quality difference kecil (<1 lap_var average), tidak signifikan untuk training. Kedua tangan punya quality yang comparable.

---

## Preprocessing Pipeline

### Configuration

```python
PalmPreprocessingConfig(
    roi_size=384,
    final_size=224,
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    profile='dataset_v3',
    adaptive_roi=True,
    adaptive_roi_scale=0.9,
    palm_core_width_ratio=0.6
)
```

### Pipeline Steps

1. **Read raw grayscale image** (PNG, original size)
2. **Extract adaptive ROI** (relaxed parameters)
   - Detect palm mask
   - Compute palm core bbox
   - Weighted centroid dengan intensity
   - Extract square ROI (side ~560-800 px)
3. **Apply CLAHE** (clip=2.0, tile=8×8)
4. **Min-max normalization** (0-255)
5. **Resize to 224×224** (INTER_AREA)
6. **Save BMP** (for training) + **JSON metadata** (for debugging & OOD detection)

### Metadata JSON Content

Each preprocessed image has JSON sidecar with:
- `subject_id`: "835" or "836"
- `distance_cm`: "22cm", "25cm", etc.
- `roi_side`: ROI size in pixels (important for OOD detection!)
- `quality`: Quality metrics (mean, std, laplacian_var)
- `quality_filter`: Pass/fail quality assessment

---

## Next Steps

### 1. ✅ Dataset Acquisition & Preprocessing — DONE
- [x] Tangan kiri (835): 50 images
- [x] Tangan kanan (836): 50 images
- [x] Select 10 best per jarak
- [x] Preprocess to 224×224 BMP
- [x] Total: 100 preprocessed images

### 2. 🔜 Build Split File — NEXT

```bash
python3 build_multi_distance_split.py \
    --dataset-root dataset_multi_distance \
    --output-file dataset_multi_distance/split_info.json \
    --subjects 835 836 \
    --train-ratio 0.6 \
    --val-ratio 0.2 \
    --test-ratio 0.2 \
    --seed 42
```

**Expected split:**
- Train: 60 images (30 per subject, ~6 per distance per subject)
- Val: 20 images (10 per subject, ~2 per distance per subject)
- Test: 20 images (10 per subject, ~2 per distance per subject)

### 3. 🔜 Training (Task 7) — After Split

```bash
python3 retrain_run7_robust.py \
    --split-file dataset_multi_distance/split_info.json \
    --augmentation-policy v2_multi_distance \
    --hand-pair-margin-loss \
    --epochs 100
```

**Expected outcome:**
- Test accuracy: 90-94% (TA-2 revised target)
- Cross-hand confusion: 0 (TA-3)
- Training time: ~2-3 hours

---

## Commands Used

### Selection Commands

```bash
# Tangan kanan (836)
python3 select_best_raw_images.py \
    --dataset-root dataset_multi_distance/836 \
    --output-dir dataset_multi_distance/836/final_raw \
    --samples-per-distance 10

# Tangan kiri (835)
python3 select_best_raw_images.py \
    --dataset-root dataset_multi_distance/835 \
    --output-dir dataset_multi_distance/835/final_raw \
    --samples-per-distance 10
```

### Preprocessing Commands

```bash
# Tangan kanan (836)
python3 preprocess_multi_distance_dataset.py \
    --input-root dataset_multi_distance/836/final_raw \
    --output-root dataset_multi_distance/836/final \
    --subject-id 836

# Tangan kiri (835)
python3 preprocess_multi_distance_dataset.py \
    --input-root dataset_multi_distance/835/final_raw \
    --output-root dataset_multi_distance/835/final \
    --subject-id 835
```

### Verification Commands

```bash
# Check dataset structure
ls -la dataset_multi_distance/

# Count BMP files per subject
for subject in 835 836; do
    echo "Subject $subject:"
    for dir in 22cm 25cm 27cm 30cm 32cm; do
        echo "  $dir: $(ls dataset_multi_distance/$subject/final/$dir/*.bmp | wc -l) BMP"
    done
done

# Verify metadata
cat dataset_multi_distance/835/final/22cm/*_preprocess.json | grep subject_id
cat dataset_multi_distance/836/final/22cm/*_preprocess.json | grep subject_id
```

---

## Files Generated

### Scripts
1. ✅ `select_best_raw_images.py` — Select top N images per distance
2. ✅ `preprocess_multi_distance_dataset.py` — Preprocess with adaptive ROI
3. ✅ `build_multi_distance_split.py` — Build train/val/test split
4. ✅ `fix_subject_id_metadata.py` — Fix subject_id in JSON metadata

### Dataset Files
1. ✅ `dataset_multi_distance/835/final_raw/` — 50 selected raw PNG (left hand)
2. ✅ `dataset_multi_distance/835/final/` — 50 preprocessed BMP + JSON (left hand)
3. ✅ `dataset_multi_distance/836/final_raw/` — 50 selected raw PNG (right hand)
4. ✅ `dataset_multi_distance/836/final/` — 50 preprocessed BMP + JSON (right hand)

### Reports
1. ✅ `DATASET_ANALYSIS_REPORT.md` — Initial analysis (tangan kanan)
2. ✅ `SELECTION_REPORT.md` — Selection process (tangan kanan)
3. ✅ `PREPROCESSING_REPORT.md` — Preprocessing details (tangan kanan)
4. ✅ `CORRECTION_REPORT.md` — Subject ID correction (835 ↔ 836)
5. ✅ `DATASET_COMPLETE_REPORT.md` — This report (final summary)

---

## Summary

✅ **Dataset acquisition & preprocessing COMPLETE**
- 100 images total (50 per tangan, 10 per jarak)
- Quality bagus (average lap_var: 15.9 ± 3.8)
- Balanced distribution (10 images per distance per subject)
- Metadata correct (subject_id: 835 = left, 836 = right)

🔜 **Next action: Build split file**
```bash
python3 build_multi_distance_split.py \
    --dataset-root dataset_multi_distance \
    --output-file dataset_multi_distance/split_info.json \
    --subjects 835 836 \
    --train-ratio 0.6 \
    --val-ratio 0.2 \
    --test-ratio 0.2 \
    --seed 42
```

📊 **Ready for training:**
- Expected accuracy: 90-94%
- Expected training time: 2-3 hours
- Zero cross-hand confusion (TA-3)

🚀 **All systems GO for Task 7 (retrain with augmentation v2 + hand-pair loss)!**

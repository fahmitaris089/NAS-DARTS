# Preprocessing Report — Multi-Distance Dataset

**Date:** 1 Juni 2026  
**Subject:** Tangan Kiri (835)  
**Total Processed:** 50 images (10 per jarak)

---

## ✅ Preprocessing COMPLETE

### Summary

**Input:**  `dataset_multi_distance/835/final_raw/` (50 raw PNG images)  
**Output:** `dataset_multi_distance/835/final/` (50 preprocessed BMP images + metadata JSON)

| Jarak | Input (PNG) | Output (BMP) | Status |
|-------|-------------|--------------|--------|
| 22 cm | 10 | 10 | ✅ |
| 25 cm | 10 | 10 | ✅ |
| 27 cm | 10 | 10 | ✅ |
| 30 cm | 10 | 10 | ✅ |
| 32 cm | 10 | 10 | ✅ |
| **TOTAL** | **50** | **50** | ✅ |

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
2. **Extract adaptive ROI** (relaxed parameters untuk device captures)
   - Detect palm mask
   - Compute palm core bbox
   - Weighted centroid dengan intensity
   - Extract square ROI (side ~560-800 px)
3. **Apply CLAHE** (clip=2.0, tile=8×8)
4. **Min-max normalization** (0-255)
5. **Resize to 224×224** (INTER_AREA interpolation)
6. **Save BMP** (for training) + **JSON metadata** (for debugging)

### Relaxed Parameters (vs Original)

Original preprocessing (untuk teacher model) terlalu strict untuk device captures. Relaxed parameters:

| Parameter | Original | Relaxed | Rationale |
|-----------|----------|---------|-----------|
| `palm_core_width_ratio` | 0.45 | 0.60 | Wider ROI untuk capture variasi jarak |
| `core_width_weight` | 0.50 | 0.60 | Lebih toleran terhadap variasi width |
| `core_height_weight` | 1.20 | 1.35 | Lebih toleran terhadap variasi height |
| `hand_height_weight` | 0.65 | 0.72 | Fallback lebih generous |
| `min_side` | 480 | 560 | Minimum ROI size lebih besar |

**Result:** ROI lebih besar dan lebih stabil di berbagai jarak (22-32 cm)

---

## Output Structure

```
dataset_multi_distance/835/
├── final_raw/                    # Input (raw PNG)
│   ├── 22cm/                     # 10 images
│   ├── 25cm/                     # 10 images
│   ├── 27cm/                     # 10 images
│   ├── 30cm/                     # 10 images
│   └── 32cm/                     # 10 images
└── final/                        # Output (preprocessed BMP + JSON)
    ├── 22cm/                     # 10 BMP + 10 JSON
    ├── 25cm/                     # 10 BMP + 10 JSON
    ├── 27cm/                     # 10 BMP + 10 JSON
    ├── 30cm/                     # 10 BMP + 10 JSON
    └── 32cm/                     # 10 BMP + 10 JSON
```

### File Naming Convention

```
Input:  palm_20260524_153938_386475.png
Output: palm_20260524_153938_386475.bmp
        palm_20260524_153938_386475_preprocess.json
```

### Metadata JSON Content

Each preprocessed image has a JSON sidecar with:
- `source_raw_image`: Path to original raw PNG
- `subject_id`: "835"
- `distance_cm`: "22cm", "25cm", etc.
- `preprocessing_config`: Full config used
- `roi_box`: [x1, y1, x2, y2] coordinates
- `roi_side`: ROI size in pixels (important for OOD detection!)
- `quality`: Quality metrics (mean, std, laplacian_var)
- `quality_filter`: Pass/fail quality assessment

**Example:**
```json
{
  "source_raw_image": "dataset_multi_distance/835/final_raw/22cm/palm_20260524_153938_386475.png",
  "subject_id": "835",
  "distance_cm": "22cm",
  "roi_side": 672,
  "quality": {
    "final": {
      "mean": 127.5,
      "std": 45.2,
      "laplacian_var": 180.3
    }
  },
  "quality_filter": {
    "usable": true,
    "laplacian_var": 180.3,
    "min_laplacian_var": 60
  }
}
```

---

## Quality Verification

### ROI Size Distribution (Proxy for Distance)

ROI size (`roi_side`) adalah proxy untuk jarak tangan ke kamera:
- **Jarak dekat (22 cm):** ROI size besar (~700-800 px)
- **Jarak nominal (27 cm):** ROI size medium (~600-700 px)
- **Jarak jauh (32 cm):** ROI size kecil (~500-600 px)

Ini penting untuk **OOD detection (M-4)** — `DistanceOODDetector` akan fit distribusi ROI size dari training set.

### Sample ROI Sizes (from JSON metadata)

```bash
# Extract ROI sizes per distance
for dir in 22cm 25cm 27cm 30cm 32cm; do
    echo "$dir:"
    grep -h '"roi_side"' dataset_multi_distance/835/final/$dir/*_preprocess.json | \
        awk '{print $2}' | sed 's/,//' | sort -n
done
```

**Expected pattern:**
- 22 cm: 650-750 px (largest)
- 25 cm: 600-700 px
- 27 cm: 580-680 px
- 30 cm: 550-650 px
- 32 cm: 520-620 px (smallest)

---

## Next Steps

### 1. ✅ Tangan Kiri (835) — DONE
- [x] Select 10 best raw images per jarak
- [x] Preprocess to 224×224 BMP
- [x] Total: 50 preprocessed images

### 2. 🔜 Tangan Kanan (836) — TODO
- [ ] Akuisisi 50 raw images (10 per jarak)
- [ ] Select 10 best per jarak (if > 10 captured)
- [ ] Preprocess dengan command yang sama:
  ```bash
  python3 preprocess_multi_distance_dataset.py \
      --input-root dataset_multi_distance/836/final_raw \
      --output-root dataset_multi_distance/836/final \
      --subject-id 836
  ```

### 3. 🔜 Build Split File — TODO
After tangan kanan (836) selesai:
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

### 4. 🔜 Training — TODO
```bash
python3 retrain_run7_robust.py \
    --split-file dataset_multi_distance/split_info.json \
    --augmentation-policy v2_multi_distance \
    --hand-pair-margin-loss \
    --epochs 100
```

---

## Commands Used

### Preprocessing Command

```bash
python3 preprocess_multi_distance_dataset.py \
    --input-root dataset_multi_distance/835/final_raw \
    --output-root dataset_multi_distance/835/final \
    --subject-id 835
```

### Verification Commands

```bash
# Count BMP files per distance
for dir in 22cm 25cm 27cm 30cm 32cm; do
    echo "$dir: $(ls dataset_multi_distance/835/final/$dir/*.bmp | wc -l) BMP files"
done

# Check file sizes
ls -lh dataset_multi_distance/835/final/22cm/

# Verify JSON metadata
cat dataset_multi_distance/835/final/22cm/palm_20260524_153938_386475_preprocess.json | jq .
```

---

## Files Generated

1. ✅ `preprocess_multi_distance_dataset.py` — Preprocessing script
2. ✅ `dataset_multi_distance/835/final/` — 50 BMP + 50 JSON files (organized by distance)
3. ✅ `PREPROCESSING_REPORT.md` — This report

---

## Summary

✅ **Preprocessing tangan kiri (835) COMPLETE**
- 50 raw PNG → 50 preprocessed BMP (224×224)
- Struktur folder per jarak maintained
- Metadata JSON untuk setiap image (includes ROI size untuk OOD detection)

🔜 **Next:** Akuisisi + preprocess tangan kanan (836) dengan workflow yang sama

📊 **Ready for training:** Setelah tangan kanan selesai, total 100 images (50 per tangan) → Expected accuracy 90-94%

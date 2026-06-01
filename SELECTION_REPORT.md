# Raw Image Selection Report — Tangan Kiri (835)

**Tanggal:** 1 Juni 2026  
**Total Selected:** 50 images (10 per jarak)  
**Output Directory:** `dataset_multi_distance/835/final_raw/`

---

## Summary

✅ **Berhasil memilih 10 raw images terbaik dari tiap jarak** berdasarkan laplacian variance (quality metric).

### Distribusi Final

| Jarak | Available | Selected | Quality (Lap Var) | Status |
|-------|-----------|----------|-------------------|--------|
| 22 cm | 12 | 10 | 20.8 ± 2.1 | ✅ |
| 25 cm | 15 | 10 | 20.0 ± 3.1 | ✅ |
| 27 cm | 11 | 10 | 16.0 ± 3.1 | ✅ |
| 30 cm | 18 | 10 | 13.6 ± 1.1 | ✅ |
| 32 cm | 14 | 10 | 11.1 ± 1.5 | ✅ |
| **TOTAL** | **70** | **50** | **16.3 ± 4.2** | ✅ |

---

## Selection Criteria

**Metric:** Laplacian Variance (proxy untuk sharpness/quality)
- Higher value = sharper image = better quality
- Script otomatis memilih top 10 images dengan laplacian variance tertinggi per jarak

**Rejected images:**
- 22 cm: 2 images dengan laplacian var terendah
- 25 cm: 5 images dengan laplacian var terendah
- 27 cm: 1 image dengan laplacian var terendah
- 30 cm: 8 images dengan laplacian var terendah
- 32 cm: 4 images dengan laplacian var terendah

**Total rejected:** 20 images (28.6% dari available)

---

## Quality Analysis per Jarak

### 22 cm (10 selected dari 12 available)
- **Quality range:** 17.3 - 24.0 (lap var)
- **Mean quality:** 20.8 ± 2.1
- **Best image:** `palm_20260524_153938_386475.png` (lap_var: 24.0)
- **Status:** ✅ Excellent quality, semua > 17

### 25 cm (10 selected dari 15 available)
- **Quality range:** 16.7 - 25.6 (lap var)
- **Mean quality:** 20.0 ± 3.1
- **Best image:** `palm_20260601_211906_016149.png` (lap_var: 25.6)
- **Status:** ✅ Excellent quality, semua > 16

### 27 cm (10 selected dari 11 available)
- **Quality range:** 10.4 - 20.1 (lap var)
- **Mean quality:** 16.0 ± 3.1
- **Best image:** `palm_20260524_161726_228260.png` (lap_var: 20.1)
- **Status:** ✅ Good quality, ada beberapa di range 10-12 (acceptable)

### 30 cm (10 selected dari 18 available)
- **Quality range:** 12.0 - 15.0 (lap var)
- **Mean quality:** 13.6 ± 1.1
- **Best image:** `palm_20260601_171440_368038.png` (lap_var: 15.0)
- **Status:** ✅ Good quality, konsisten (std kecil)

### 32 cm (10 selected dari 14 available)
- **Quality range:** 9.4 - 13.5 (lap var)
- **Mean quality:** 11.1 ± 1.5
- **Best image:** `palm_20260601_172201_127176.png` (lap_var: 13.5)
- **Status:** ✅ Acceptable quality, jarak terjauh jadi lap var lebih rendah (expected)

---

## Observations

### ✅ Good News

1. **Balanced distribution:** Semua jarak punya exactly 10 samples
2. **Quality gradient expected:** Laplacian variance menurun dari 22 cm (20.8) ke 32 cm (11.1), yang expected karena jarak lebih jauh = intensitas NIR lebih rendah
3. **Consistent quality:** Std deviation kecil di 30 cm (1.1) dan 32 cm (1.5), menunjukkan capture consistency bagus
4. **All above threshold:** Semua selected images punya laplacian var > 9, yang masih acceptable untuk preprocessing

### 📊 Quality Trend

```
Quality (Laplacian Variance) vs Distance:
22 cm: ████████████████████ 20.8
25 cm: ███████████████████  20.0
27 cm: ███████████████      16.0
30 cm: █████████████        13.6
32 cm: ███████████          11.1
```

**Interpretation:**
- Jarak dekat (22-25 cm): Quality tinggi (20+), palm vein pattern sangat jelas
- Jarak nominal (27 cm): Quality good (16), baseline training
- Jarak jauh (30-32 cm): Quality acceptable (11-14), masih cukup untuk model belajar

---

## Next Steps

### 1. ✅ Tangan Kiri (835) — DONE
- [x] Pilih 10 raw images terbaik per jarak
- [x] Copy ke `dataset_multi_distance/835/final_raw/`
- [x] Total: 50 samples

### 2. 🔜 Tangan Kanan (836) — TODO
- [ ] Akuisisi 50 raw images (10 per jarak: 22, 25, 27, 30, 32 cm)
- [ ] Copy ke `dataset_multi_distance/836/final_raw/`
- [ ] **Estimated effort:** 45-60 menit

### 3. 🔜 Preprocessing — TODO
- [ ] Run preprocessing script untuk convert raw → final (224×224)
- [ ] Apply profile `dataset_v3` (adaptive ROI, CLAHE, etc.)
- [ ] Output: `dataset_multi_distance/{835,836}/final/`

### 4. 🔜 Training — TODO
- [ ] Build split file (60% train, 20% val, 20% test)
- [ ] Run Task 7: Retrain dengan augmentation v2 + hand-pair loss
- [ ] Expected accuracy: 90-94%

---

## File Structure

```
dataset_multi_distance/835/
├── final_raw/                    # ✅ NEW: Selected raw images
│   ├── 22cm/                     # 10 images
│   ├── 25cm/                     # 10 images
│   ├── 27cm/                     # 10 images
│   ├── 30cm/                     # 10 images
│   ├── 32cm/                     # 10 images
│   └── selection_summary.json    # Selection metadata
├── 22cm/
│   ├── raw/                      # Original captures (12 images)
│   ├── processed/
│   ├── final/                    # Preprocessed (12 images)
│   └── visualizations/
├── 25cm/
│   ├── raw/                      # Original captures (15 images)
│   └── ...
└── ...
```

---

## Commands Used

```bash
# Select 10 best raw images per distance
python3 select_best_raw_images.py \
    --dataset-root dataset_multi_distance/835 \
    --output-dir dataset_multi_distance/835/final_raw \
    --samples-per-distance 10

# Verify selection
find dataset_multi_distance/835/final_raw -type f -name "*.png" | wc -l
# Output: 50

# Count per distance
for dir in 22cm 25cm 27cm 30cm 32cm; do
    echo "$dir: $(ls dataset_multi_distance/835/final_raw/$dir/*.png | wc -l) files"
done
# Output: 10 files each
```

---

## Metadata

Selection summary tersimpan di:
- `dataset_multi_distance/835/final_raw/selection_summary.json`

Contains:
- List of selected files per distance
- Laplacian variance scores
- Quality statistics (mean, std, min, max)
- Ranking (best to worst)

---

## Recommendations

### For Tangan Kanan (836) Acquisition

Gunakan command yang sama untuk capture:
```bash
python3 capture_on_hand_detect.py \
    --size 1920x1080 \
    --fps 30 \
    --exposure-us 8000 \
    --gain 1.1 \
    --awbgains 1.0,1.0 \
    --brightness -0.04 \
    --contrast 1.3 \
    --saturation 0 \
    --out-dir dataset_multi_distance/836/{distance}cm \
    --stable-frames 12 \
    --burst-frames 10 \
    --preprocess \
    --preprocess-profile dataset_v3 \
    --quality-filter \
    --quality-min-laplacian-var 60 \
    --save-rejected
```

**Target per jarak:** 10-12 captures (untuk buffer, nanti pilih 10 terbaik)

**Estimated time:**
- Per jarak: 8-10 menit (10 captures × ~1 menit per capture)
- Total 5 jarak: 40-50 menit

---

## Conclusion

✅ **Tangan kiri (835) dataset ready:** 50 raw images terpilih (10 per jarak) dengan quality bagus

🔜 **Next action:** Akuisisi tangan kanan (836) dengan target yang sama (50 images, 10 per jarak)

📊 **Expected final dataset:** 100 images total (50 per tangan, 10 per jarak) → Training accuracy 90-94%

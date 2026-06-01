# Correction Report — Subject ID Fix

**Date:** 1 Juni 2026  
**Issue:** Dataset yang sudah diakuisisi adalah **tangan KANAN (836)**, bukan tangan kiri (835)

---

## ✅ Correction COMPLETE

### Changes Made

1. **Renamed folder:** `dataset_multi_distance/835/` → `dataset_multi_distance/836/`
2. **Updated metadata:** 50 JSON files, subject_id changed from "835" to "836"

### Verification

```bash
# Check folder structure
ls -la dataset_multi_distance/
# Output: 836/ (correct)

# Check metadata
cat dataset_multi_distance/836/final/22cm/palm_20260524_153938_386475_preprocess.json | grep subject_id
# Output: "subject_id": "836" (correct)
```

---

## Current Status

### ✅ Tangan Kanan (836) — COMPLETE

**Dataset:** `dataset_multi_distance/836/`

| Jarak | Raw (final_raw) | Preprocessed (final) | Status |
|-------|-----------------|----------------------|--------|
| 22 cm | 10 PNG | 10 BMP + 10 JSON | ✅ |
| 25 cm | 10 PNG | 10 BMP + 10 JSON | ✅ |
| 27 cm | 10 PNG | 10 BMP + 10 JSON | ✅ |
| 30 cm | 10 PNG | 10 BMP + 10 JSON | ✅ |
| 32 cm | 10 PNG | 10 BMP + 10 JSON | ✅ |
| **TOTAL** | **50 PNG** | **50 BMP + 50 JSON** | ✅ |

**Subject ID:** 836 (tangan kanan) ✅  
**Metadata:** All JSON files updated ✅

---

### 🔜 Tangan Kiri (835) — TODO

**Target:** 50 images (10 per jarak: 22, 25, 27, 30, 32 cm)  
**Estimated time:** 45-60 menit

**Workflow:**
1. Akuisisi 50 raw images untuk tangan kiri
2. Select 10 best per jarak (jika > 10 captured)
3. Preprocess:
   ```bash
   python3 preprocess_multi_distance_dataset.py \
       --input-root dataset_multi_distance/835/final_raw \
       --output-root dataset_multi_distance/835/final \
       --subject-id 835
   ```

---

## Updated Dataset Structure

```
dataset_multi_distance/
├── 836/                        # ✅ Tangan KANAN (DONE)
│   ├── final_raw/              # 50 raw PNG (10 per jarak)
│   │   ├── 22cm/ (10)
│   │   ├── 25cm/ (10)
│   │   ├── 27cm/ (10)
│   │   ├── 30cm/ (10)
│   │   └── 32cm/ (10)
│   └── final/                  # 50 preprocessed BMP + JSON
│       ├── 22cm/ (10 BMP + 10 JSON)
│       ├── 25cm/ (10 BMP + 10 JSON)
│       ├── 27cm/ (10 BMP + 10 JSON)
│       ├── 30cm/ (10 BMP + 10 JSON)
│       └── 32cm/ (10 BMP + 10 JSON)
└── 835/                        # 🔜 Tangan KIRI (TODO)
    ├── final_raw/              # (belum ada)
    └── final/                  # (belum ada)
```

---

## Next Steps

### 1. 🔜 Akuisisi Tangan Kiri (835)

**Command untuk capture (per jarak):**
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
    --out-dir dataset_multi_distance/835/{distance}cm \
    --stable-frames 12 \
    --burst-frames 10 \
    --preprocess \
    --preprocess-profile dataset_v3 \
    --quality-filter \
    --quality-min-laplacian-var 60 \
    --save-rejected
```

**Target per jarak:** 10-12 captures (untuk buffer, nanti pilih 10 terbaik)

### 2. 🔜 Select Best Raw Images (Tangan Kiri)

```bash
python3 select_best_raw_images.py \
    --dataset-root dataset_multi_distance/835 \
    --output-dir dataset_multi_distance/835/final_raw \
    --samples-per-distance 10
```

### 3. 🔜 Preprocess (Tangan Kiri)

```bash
python3 preprocess_multi_distance_dataset.py \
    --input-root dataset_multi_distance/835/final_raw \
    --output-root dataset_multi_distance/835/final \
    --subject-id 835
```

### 4. 🔜 Build Split File (Kedua Tangan)

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

### 5. 🔜 Training (Task 7)

```bash
python3 retrain_run7_robust.py \
    --split-file dataset_multi_distance/split_info.json \
    --augmentation-policy v2_multi_distance \
    --hand-pair-margin-loss \
    --epochs 100
```

---

## Summary

✅ **Correction DONE:**
- Folder renamed: 835 → 836
- Metadata updated: 50 JSON files (subject_id: "835" → "836")
- Dataset tangan kanan (836) sudah complete dan correct

🔜 **Next action:**
- Akuisisi tangan kiri (835) dengan workflow yang sama
- Target: 50 images (10 per jarak)
- Estimated time: 45-60 menit

📊 **Final target:**
- 100 images total (50 per tangan, 10 per jarak)
- Expected training accuracy: 90-94%
- Zero cross-hand confusion (TA-3)

# Dataset Analysis Results - Palm Vein Multi-Distance

**Date:** June 1, 2026  
**User ID:** 835 (Left Hand)  
**Total Images:** 63 across 5 distances

---

## 📊 Summary

Your dataset has **good image quality** but **insufficient quantity** for robust cross-distance recognition.

### Current Status:

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Total images | 63 | 83 | 🔴 Need +20 |
| Avg per distance | 12.6 | 15-20 | 🔴 Below target |
| Image sharpness | 149.42 | >100 | ✅ Excellent |
| Vein visibility | 0.020 | >0.015 | ⚠️ Marginal |

---

## 🎯 Root Cause of Low Robustness

Based on analysis, your model's lack of robustness is caused by:

### 1. **Insufficient Training Data** (PRIMARY)
- Original training: 10 images at 27cm only
- Current dataset: 11 images at 27cm (minimal improvement)
- **Impact:** Model has no experience with distance variations

### 2. **Unbalanced Distribution**
- 25cm: only 8 images (weakest link)
- 30cm: 18 images (over-represented)
- **Impact:** Model biased toward 30cm, poor at 25cm

### 3. **Weak Vein Patterns at Close Distances**
- 22cm edge_density: 0.0053 (target: >0.015)
- 25cm edge_density: 0.0078 (target: >0.015)
- **Impact:** Model can't extract features at close range

### 4. **Training Pipeline Issues** (from bugfix.md)
- `RandomHorizontalFlip` confuses left vs right hand
- No scale augmentation for distance variations
- **Impact:** Model doesn't learn distance-invariant features

---

## 🔧 Action Plan

### Phase 1: Expand Dataset (IMMEDIATE)

**Capture +20 images with this priority:**

```
Priority 1: 25cm → +10 images (8 → 18)  [CRITICAL]
Priority 2: 27cm → +9 images  (11 → 20) [TRAINING CENTER]
Priority 3: 22cm → +3 images  (12 → 15) [BOUNDARY]
Priority 4: 32cm → +1 image   (14 → 15) [BOUNDARY]
```

**Use these commands:**

For 22-25cm (adjust for better vein visibility):
```bash
python3 capture_on_hand_detect.py \
  --size 1920x1080 --fps 30 \
  --exposure-us 6000 --gain 1.0 \
  --contrast 1.5 --brightness -0.04 \
  --out-dir dataset_multi_distance/835/25cm \
  --stable-frames 12 --burst-frames 10 \
  --preprocess --preprocess-profile dataset_v3 \
  --quality-filter --quality-min-laplacian-var 60
```

For 27-32cm (current settings work):
```bash
python3 capture_on_hand_detect.py \
  --size 1920x1080 --fps 30 \
  --exposure-us 8000 --gain 1.1 \
  --contrast 1.3 --brightness -0.04 \
  --out-dir dataset_multi_distance/835/27cm \
  --stable-frames 12 --burst-frames 10 \
  --preprocess --preprocess-profile dataset_v3 \
  --quality-filter --quality-min-laplacian-var 60
```

**Validation during capture:**
```bash
# Check individual image quality
python3 quick_validate_image.py dataset_multi_distance/835/25cm/final/latest_image.png

# Check overall dataset after session
python3 analyze_dataset_quality.py
```

### Phase 2: Fix Training Pipeline (AFTER DATASET EXPANSION)

1. **Remove horizontal flip augmentation**
   - Edit training script
   - Remove `transforms.RandomHorizontalFlip(p=0.5)`

2. **Add scale augmentation**
   ```python
   transforms.RandomAffine(
       degrees=0,
       scale=(0.85, 1.15),  # Simulates ±3cm distance variation
       translate=(0.05, 0.05)
   )
   ```

3. **Increase training epochs**
   - From 50 → 100 epochs
   - Add early stopping (patience=15)

4. **Adjust learning rate schedule**
   - Use cosine annealing
   - Warmup for first 10 epochs

### Phase 3: Retrain and Evaluate

1. **Retrain model** with 83 images + fixed augmentation
2. **Cross-distance evaluation:**
   - Enroll at 27cm
   - Test at 22cm, 25cm, 30cm, 32cm
   - Measure accuracy at each distance

3. **Expected results:**
   - Intra-distance (27cm): ≥95% accuracy
   - Cross-distance (±5cm): ≥90% accuracy
   - Boundary (22cm, 32cm): ≥85% accuracy

---

## 📈 Detailed Metrics by Distance

### 22cm (Boundary - Close)
- **Samples:** 12 → target 15 (+3 needed)
- **Sharpness:** 151.42 ± 23.57 ✅
- **Vein visibility:** 0.0053 🔴 (too low)
- **Action:** Capture +3 with lower exposure (6000us)

### 25cm (Critical Weak Point)
- **Samples:** 8 → target 18 (+10 needed) 🔴
- **Sharpness:** 152.54 ± 17.96 ✅
- **Vein visibility:** 0.0078 🔴 (too low)
- **Action:** PRIORITY - capture +10 with adjusted settings

### 27cm (Training Center)
- **Samples:** 11 → target 20 (+9 needed) 🔴
- **Sharpness:** 157.92 ± 46.56 ✅
- **Vein visibility:** 0.0193 ⚠️ (marginal)
- **Action:** Capture +9 with current settings

### 30cm (Over-represented)
- **Samples:** 18 → target 15 (reduce by 3) ✅
- **Sharpness:** 137.35 ± 28.20 ✅
- **Vein visibility:** 0.0255 ✅
- **Action:** No additional capture needed

### 32cm (Boundary - Far)
- **Samples:** 14 → target 15 (+1 needed)
- **Sharpness:** 154.74 ± 25.15 ✅
- **Vein visibility:** 0.0342 ✅ (best)
- **Action:** Capture +1 with current settings

---

## 📁 Files Generated

1. **`analyze_dataset_quality.py`** - Full dataset analysis script
2. **`quick_validate_image.py`** - Quick validation during capture
3. **`capture_guide_next_session.md`** - Step-by-step capture guide
4. **`dataset_analysis_results/`** - Analysis outputs:
   - `quality_report.json` - Detailed metrics
   - `dataset_quality_analysis.png` - Visualizations

---

## 🚀 Next Steps

### Today:
1. ✅ Dataset analysis complete
2. 🔄 Review capture guide: `capture_guide_next_session.md`
3. 🔄 Prepare capture setup (ruler, lighting, camera)

### This Week:
4. 🔄 Capture session (~30 min): +20 images
5. 🔄 Validate results: `python3 analyze_dataset_quality.py`
6. 🔄 Fix training pipeline (remove horizontal flip, add scale aug)
7. 🔄 Retrain model with balanced dataset

### Validation:
8. 🔄 Cross-distance evaluation
9. 🔄 Live enrollment + verification test
10. 🔄 Measure FAR/FRR at different distances

---

## 💡 Key Insights

1. **Your capture settings are good** - sharpness is excellent across all distances
2. **Vein visibility issue is fixable** - just need lower exposure at 22-25cm
3. **Sample size is the main bottleneck** - 63 → 83 images will significantly improve robustness
4. **Training pipeline needs fixes** - horizontal flip is harmful, scale augmentation is missing

**Bottom line:** With +20 more images and fixed augmentation, you should see cross-distance accuracy improve from ~70% to ~90%. 🎯

---

## 📞 Questions?

If you encounter issues during capture:
- Check `capture_guide_next_session.md` for troubleshooting
- Use `quick_validate_image.py` to validate individual images
- Run `analyze_dataset_quality.py` after each distance to check progress

Good luck with your capture session! 🚀

# Dataset Analysis Summary - Multi-Distance Palm Vein (User 835)

**Analysis Date:** June 1, 2026  
**Dataset Path:** `/Users/fahmitaris/Downloads/NAS-DARTS/dataset_multi_distance/835/`  
**Analysis Script:** `analyze_dataset_quality.py`

---

## Executive Summary

✅ **Good News:**
- Image sharpness is excellent (Laplacian: 149.42 ± 30.91, well above threshold of 60)
- All 5 distance variations captured (22cm, 25cm, 27cm, 30cm, 32cm)
- Total 63 images collected across distances

⚠️ **Bottlenecks Identified:**
1. **Sample size below optimal** — 12.6 avg images/distance (target: 15-20)
2. **Unbalanced distribution** — 25cm has only 8 images, 30cm has 18
3. **Low vein visibility at close distances** — 22cm and 25cm show weak vein patterns

---

## Detailed Metrics by Distance

| Distance | Samples | Sharpness (Laplacian) | Vein Visibility (Edge Density) | Status |
|----------|---------|----------------------|-------------------------------|--------|
| **22cm** | 12 | 151.42 ± 23.57 | 0.0053 ± 0.0018 | ⚠️ Low vein visibility |
| **25cm** | 8 | 152.54 ± 17.96 | 0.0078 ± 0.0020 | 🔴 Needs more samples + low visibility |
| **27cm** | 11 | 157.92 ± 46.56 | 0.0193 ± 0.0019 | ⚠️ Needs 4 more samples |
| **30cm** | 18 | 137.35 ± 28.20 | 0.0255 ± 0.0052 | ✅ Good sample count |
| **32cm** | 14 | 154.74 ± 25.15 | 0.0342 ± 0.0059 | ✅ Good (boundary distance) |

### Key Observations:

1. **Vein visibility increases with distance** — 32cm shows 6.4x better edge density than 22cm
   - This suggests close distances (22-25cm) may have overexposure or insufficient contrast
   
2. **25cm is the weakest link** — only 8 samples AND low vein visibility
   - This is critical because 25cm is close to your original training distance (27cm)
   
3. **30cm has best sample count** — 18 images, but could be reduced to balance with others

---

## Root Cause Analysis: Why Model Lacks Robustness

Based on dataset analysis + bugfix documentation, the robustness bottleneck is:

### 1. **Training Distribution Too Narrow** (PRIMARY ISSUE)
- Original model trained on **10 images at 27cm only**
- New dataset has 11 images at 27cm — minimal improvement
- Model embedding space has no margin for distance variations

### 2. **Insufficient Samples at Critical Distances**
- **25cm (8 images):** Too few to learn close-distance characteristics
- **22cm (12 images):** Boundary case needs ≥15 samples for robustness
- **27cm (11 images):** Training distance should have most samples (target: 20)

### 3. **Low Vein Visibility at Close Distances**
- 22cm and 25cm show weak vein patterns (edge_density < 0.01)
- Possible causes:
  - Overexposure at close range
  - Hand too close to IR illumination
  - Preprocessing not optimized for close distances

### 4. **Data Augmentation Issues** (from bugfix.md)
- `RandomHorizontalFlip` is still active — confuses left vs right hand
- No distance-specific augmentation to simulate scale variations

---

## Actionable Recommendations

### Priority 1: Balance and Expand Dataset (HIGH)

**Target distribution:**
```
22cm: 12 → 15 images (+3)  [boundary distance]
25cm:  8 → 18 images (+10) [critical weak point]
27cm: 11 → 20 images (+9)  [training center - needs most samples]
30cm: 18 → 15 images (-3)  [reduce to balance]
32cm: 14 → 15 images (+1)  [boundary distance]
```

**Total target: 83 images** (currently 63, need +20 more)

**Capture strategy:**
1. **Prioritize 25cm** — capture 10 more images immediately
2. **Boost 27cm** — capture 9 more images (this is your training center)
3. **Top up boundaries** — 22cm (+3), 32cm (+1)
4. **Optionally reduce 30cm** — move 3 images to other distances if needed

### Priority 2: Fix Vein Visibility at Close Distances (HIGH)

**Problem:** 22cm and 25cm have weak vein patterns

**Experiment with capture settings:**

Current settings:
```bash
--exposure-us 8000 --gain 1.1 --contrast 1.3
```

**Try for 22-25cm:**
```bash
# Option A: Reduce exposure (prevent overexposure)
--exposure-us 6000 --gain 1.0 --contrast 1.5

# Option B: Increase contrast + reduce brightness
--exposure-us 7000 --gain 1.1 --brightness -0.08 --contrast 1.5
```

**Validation:** Re-run `analyze_dataset_quality.py` after capturing 5 test images to check if edge_density improves to >0.015

### Priority 3: Fix Training Pipeline (MEDIUM)

**From bugfix.md, these issues must be fixed:**

1. **Remove `RandomHorizontalFlip`** from augmentation
   ```python
   # REMOVE THIS:
   transforms.RandomHorizontalFlip(p=0.5)
   ```

2. **Add distance-aware augmentation**
   ```python
   # Add scale augmentation to simulate distance variations
   transforms.RandomAffine(
       degrees=0,
       scale=(0.85, 1.15),  # ±15% scale = ±3cm distance variation
       translate=(0.05, 0.05)
   )
   ```

3. **Increase training epochs** — from 50 to 100 epochs with early stopping

### Priority 4: Adjust Success Criteria (MEDIUM)

**Current target (design.md):** ≥95% accuracy

**Realistic target with 63-83 images:**
- **Intra-distance accuracy:** ≥95% (same distance as enrollment)
- **Cross-distance accuracy:** ≥90% (±5cm from enrollment)
- **Boundary accuracy:** ≥85% (22cm and 32cm)

---

## Next Steps

### Immediate Actions (Today):

1. ✅ **Dataset analysis complete** — you now have metrics
2. 🔄 **Capture 10 more images at 25cm** with adjusted settings
3. 🔄 **Capture 9 more images at 27cm** (training center)

### Short-term (This Week):

4. 🔄 **Fix augmentation pipeline** — remove horizontal flip, add scale augmentation
5. 🔄 **Retrain model** with balanced dataset (83 images)
6. 🔄 **Run cross-distance evaluation** to measure robustness improvement

### Validation:

7. 🔄 **Test live enrollment + verification** at all 5 distances
8. 🔄 **Measure FAR/FRR** at different distance variations

---

## Visualization

See `dataset_analysis_results/dataset_quality_analysis.png` for:
- Sample distribution bar chart
- Sharpness boxplots per distance
- Vein visibility comparison
- Contrast ratio analysis

---

## Technical Notes

### Why 15-20 samples per distance?

- **Statistical significance:** Need ≥15 samples for reliable mean/variance estimation
- **Augmentation multiplier:** With 5x augmentation, 15 samples → 75 training examples per distance
- **Cross-validation:** 15 samples allows 3-fold CV with 5 samples per fold

### Why prioritize 27cm?

- This is your **training center distance** (closest to original 27cm training)
- Model will interpolate from this anchor point to other distances
- More samples at center = better generalization to boundaries

### Why reduce 30cm?

- Already has 18 samples (above target)
- Not a boundary distance (less critical)
- Can redistribute to more critical distances

---

## Conclusion

Your dataset quality is **good but insufficient in quantity**. The primary bottleneck is:

1. **Too few samples** — especially at 25cm (8) and 27cm (11)
2. **Weak vein visibility at close distances** — needs capture setting adjustment
3. **Training pipeline issues** — horizontal flip must be removed

**Minimum viable action:** Capture +10 images at 25cm and +9 at 27cm, then retrain with fixed augmentation.

**Expected improvement:** Cross-distance accuracy should improve from ~70% to ~90% with these changes.

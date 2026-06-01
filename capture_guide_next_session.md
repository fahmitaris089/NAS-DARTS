# Quick Capture Guide - Next Session

## Target: Capture +20 images to balance dataset

### Priority Order:

1. **25cm: +10 images** (CRITICAL - currently only 8)
2. **27cm: +9 images** (TRAINING CENTER - needs most samples)
3. **22cm: +3 images** (boundary distance)
4. **32cm: +1 image** (boundary distance)

---

## Capture Commands by Distance

### For 22cm and 25cm (close distances - adjust for better vein visibility):

```bash
# Option A: Lower exposure
python3 capture_on_hand_detect.py \
  --size 1920x1080 \
  --fps 30 \
  --exposure-us 6000 \
  --gain 1.0 \
  --awbgains 1.0,1.0 \
  --brightness -0.04 \
  --contrast 1.5 \
  --saturation 0 \
  --out-dir dataset_multi_distance/835/25cm \
  --stable-frames 12 \
  --burst-frames 10 \
  --preprocess \
  --preprocess-profile dataset_v3 \
  --quality-filter \
  --quality-min-laplacian-var 60 \
  --save-rejected
```

### For 27cm, 30cm, 32cm (current settings work well):

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
  --out-dir dataset_multi_distance/835/27cm \
  --stable-frames 12 \
  --burst-frames 10 \
  --preprocess \
  --preprocess-profile dataset_v3 \
  --quality-filter \
  --quality-min-laplacian-var 60 \
  --save-rejected
```

---

## Capture Checklist

### Before Each Distance:

- [ ] Measure distance with ruler (camera to palm center)
- [ ] Adjust `--out-dir` to correct distance folder
- [ ] For 22-25cm: use exposure-us 6000, contrast 1.5
- [ ] For 27-32cm: use exposure-us 8000, contrast 1.3

### During Capture:

- [ ] Keep hand stable for 12 frames
- [ ] Slight variations in hand position (±2cm horizontal/vertical)
- [ ] Slight variations in hand rotation (±5 degrees)
- [ ] Ensure palm vein pattern is visible in preview

### After Each Burst:

- [ ] Check `final/` folder for accepted images
- [ ] Verify Laplacian variance >60 in filename
- [ ] If rejected, adjust hand position and retry

---

## Quick Validation After Capture

Run this after each distance to check quality:

```bash
python3 analyze_dataset_quality.py
```

**Target metrics:**
- Laplacian variance: >100 (sharpness)
- Edge density: >0.015 (vein visibility)
- Sample count: 15-20 per distance

---

## Session Plan (Estimated 30 minutes)

1. **25cm: 10 images** (~10 min)
   - Use adjusted settings (exposure 6000, contrast 1.5)
   - Validate first 2 images before continuing
   
2. **27cm: 9 images** (~8 min)
   - Use standard settings
   - This is training center - prioritize quality
   
3. **22cm: 3 images** (~3 min)
   - Use adjusted settings
   
4. **32cm: 1 image** (~1 min)
   - Use standard settings

5. **Validation** (~5 min)
   - Run `analyze_dataset_quality.py`
   - Check visualization
   - Verify target counts reached

---

## After Capture Session

1. Run full analysis:
   ```bash
   python3 analyze_dataset_quality.py
   ```

2. Check results in `dataset_analysis_results/`:
   - `quality_report.json` - detailed metrics
   - `dataset_quality_analysis.png` - visualizations

3. If metrics look good, proceed to retrain:
   - Fix augmentation (remove horizontal flip)
   - Add scale augmentation
   - Train for 100 epochs

---

## Troubleshooting

### If vein pattern not visible at 22-25cm:

Try even lower exposure:
```bash
--exposure-us 5000 --gain 0.9 --contrast 1.6
```

### If images too dark at 32cm:

Increase exposure:
```bash
--exposure-us 9000 --gain 1.2
```

### If hand detection fails:

- Increase brightness: `--brightness 0.0`
- Check lighting conditions
- Ensure hand fills 60-80% of frame

---

## Success Criteria

After this session, you should have:

- ✅ 83 total images (currently 63)
- ✅ 15-20 images per distance
- ✅ Edge density >0.015 for all distances
- ✅ Balanced distribution (max difference <5 images)

Then you're ready to retrain with improved robustness! 🚀

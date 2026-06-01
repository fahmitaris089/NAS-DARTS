# Implementation Priority - Based on Dataset Analysis

**Date:** June 1, 2026  
**Context:** Dataset analysis reveals 63 images across 5 distances (22-32cm) with good quality but insufficient quantity

---

## Critical Path Update

Based on the dataset analysis in `dataset_analysis_summary.md`, we need to **prioritize data collection before training**. The current 63 images are insufficient for robust cross-distance recognition.

### Revised Implementation Order:

## Phase 1: Data Collection (IMMEDIATE - BLOCKING)

**Status:** 🔴 **MUST COMPLETE BEFORE TRAINING**

### Task 6.1 - Operator-driven data collection (MODIFIED)

**Original plan:** Collect 10 samples per distance (50 total)  
**Revised plan:** Collect to reach 15-20 samples per distance (83 total)

**Current state:**
```
22cm: 12 images → target 15 (+3 needed)
25cm:  8 images → target 18 (+10 needed) ⚠️ CRITICAL
27cm: 11 images → target 20 (+9 needed)  ⚠️ TRAINING CENTER
30cm: 18 images → target 15 (already sufficient)
32cm: 14 images → target 15 (+1 needed)
```

**Action items:**

1. **Capture +20 images** using `capture_guide_next_session.md`
   - Priority 1: 25cm (+10 images) - currently weakest
   - Priority 2: 27cm (+9 images) - training center needs most samples
   - Priority 3: 22cm (+3 images) - boundary distance
   - Priority 4: 32cm (+1 image) - boundary distance

2. **Adjust capture settings for close distances (22-25cm)**
   - Current settings cause overexposure → weak vein patterns
   - Use `--exposure-us 6000 --gain 1.0 --contrast 1.5` for 22-25cm
   - Keep `--exposure-us 8000 --gain 1.1 --contrast 1.3` for 27-32cm

3. **Validate during capture**
   ```bash
   # After each burst
   python3 quick_validate_image.py dataset_multi_distance/835/25cm/final/latest.png
   
   # After each distance
   python3 analyze_dataset_quality.py
   ```

4. **Build held-out test split**
   - Reserve 5 samples per (subject, distance) = 25 images for test
   - Remaining 58 images for train/val
   - Save manifest as `dataset_multi_distance/manifest.json`

**Estimated time:** 30-45 minutes  
**Blocking:** Tasks 7, 7.1, 8, 8.1, 9, 9.1, 10, 11, 12, 13

---

## Phase 2: Foundation (Parallel - After Data Collection)

**Status:** ⚠️ Ready to start in parallel once data collection completes

### Wave 1: Independent Module Edits

These can run in parallel:

- [ ] **Task 1 + 1.1** - Add `palm_core_side_px` alias (M-3) + determinism test (VP-1)
  - Zero-risk metadata addition
  - Estimated: 2 hours
  
- [ ] **Task 2 + 2.1** - Implement `DistanceOODDetector` (M-4) + property tests
  - Standalone module with hypothesis tests
  - Estimated: 3 hours
  
- [ ] **Task 3 + 3.1** - Augmentation policy v2 flag (M-2) + unit test
  - **CRITICAL FIX:** Removes `RandomHorizontalFlip` that confuses left/right hands
  - Adds scale augmentation to simulate distance variations
  - Estimated: 2 hours
  
- [ ] **Task 4 + 4.1** - Hand-pair margin loss flag (M-5) + unit test
  - Prevents cross-hand confusion
  - Estimated: 2 hours

**Total Wave 1 time:** 3 hours (if parallelized) or 9 hours (if sequential)

---

## Phase 3: Integration (After Wave 1)

### Wave 2: Wire Components

- [ ] **Task 5 + 5.1** - Wire OOD detector into decision rule (M-8) + baseline preservation test (VP-2)
  - Depends on: Tasks 1, 2
  - Estimated: 2 hours

- [ ] **Task 6** - Multi-distance collection wrapper (M-1)
  - Depends on: Task 1
  - **NOTE:** This is now mostly complete via manual capture, but wrapper script still useful for future
  - Estimated: 1 hour

**Total Wave 2 time:** 2 hours (if parallelized) or 3 hours (if sequential)

---

## Phase 4: Training (COMPUTE-INTENSIVE)

### Wave 3: Retrain with Fixes

- [ ] **Task 7** - Retrain NAS-DARTS with v2 augmentation + hand-pair loss
  - Depends on: Tasks 3, 4, 6.1 (data collection)
  - **CRITICAL:** This is where the robustness improvement happens
  - Uses `augmentation_policy="v2_multi_distance"` (no horizontal flip, scale aug)
  - Uses `hand_pair_margin_loss=True` (cross-hand separation)
  - Estimated: 4-6 hours (GPU training time)

- [ ] **Task 7.1** - Fit `DistanceOODDetector` from training set
  - Depends on: Task 7
  - Estimated: 30 minutes

**Total Wave 3 time:** 4-6 hours (mostly GPU time)

---

## Phase 5: Deployment Prep (After Training)

### Wave 4: Enrollment & Calibration

- [ ] **Task 8 + 8.1** - Multi-distance enrollment (M-6) + generate v2 template store
  - Depends on: Task 7
  - Estimated: 1 hour

- [ ] **Task 9 + 9.1** - Threshold calibration (M-7) + run calibration
  - Depends on: Tasks 7.1, 8.1
  - Sweeps grid to find optimal operating point
  - Estimated: 1 hour

**Total Wave 4 time:** 1 hour (if parallelized) or 2 hours (if sequential)

---

## Phase 6: Verification (Final Gate)

### Wave 5: Property Verification

- [ ] **Task 10** - Verify multi-distance accuracy (VP-3)
  - Target: ≥95% accuracy on D_validated
  - Depends on: Task 9.1
  - Estimated: 30 minutes

- [ ] **Task 11** - Verify cross-hand zero false-accept (VP-4)
  - Target: 0% false-accept on cross-hand pairs
  - Depends on: Task 9.1
  - Estimated: 30 minutes

- [ ] **Task 12** - Verify OOD reject at 18cm and 38cm (VP-5)
  - Requires additional capture session (20 images at 18cm, 20 at 38cm)
  - Target: ≥90% rejection rate
  - Depends on: Tasks 6.1, 9.1
  - Estimated: 1 hour capture + 30 minutes verification

**Total Wave 5 time:** 2 hours

### Wave 6: Final Acceptance

- [ ] **Task 13** - Property-based verification harness (VP-6)
  - Aggregates VP-1 through VP-5
  - Final acceptance gate
  - Estimated: 2 hours

- [ ] **Task 14** - Documentation and rollout notes
  - Update protocol documentation
  - Estimated: 1 hour

**Total Wave 6 time:** 3 hours

---

## Total Timeline Estimate

| Phase | Time | Blocking? |
|-------|------|-----------|
| Phase 1: Data Collection | 0.5-1 hour | 🔴 YES - blocks training |
| Phase 2: Foundation (Wave 1) | 3 hours (parallel) | ⚠️ Blocks integration |
| Phase 3: Integration (Wave 2) | 2 hours (parallel) | ⚠️ Blocks training |
| Phase 4: Training (Wave 3) | 4-6 hours (GPU) | 🔴 YES - blocks deployment |
| Phase 5: Deployment (Wave 4) | 1 hour (parallel) | ⚠️ Blocks verification |
| Phase 6: Verification (Wave 5-6) | 5 hours | Final gate |
| **Total (critical path)** | **15-18 hours** | |
| **Total (if parallelized)** | **12-15 hours** | |

---

## Immediate Next Steps (Today)

### Step 1: Complete Data Collection (30-45 min)

```bash
# Follow the guide
cat capture_guide_next_session.md

# Capture priority order:
# 1. 25cm: +10 images (use adjusted settings)
# 2. 27cm: +9 images (use standard settings)
# 3. 22cm: +3 images (use adjusted settings)
# 4. 32cm: +1 image (use standard settings)

# Validate after each distance
python3 analyze_dataset_quality.py
```

### Step 2: Build Test Split (15 min)

```bash
# Reserve 5 samples per distance for test
# Create manifest.json
# Document split strategy
```

### Step 3: Start Wave 1 Tasks (3 hours)

Once data collection is complete, start these in parallel:
- Task 1 + 1.1 (palm_core_side_px alias)
- Task 2 + 2.1 (OOD detector module)
- Task 3 + 3.1 (augmentation policy v2) ← **HIGHEST IMPACT**
- Task 4 + 4.1 (hand-pair margin loss)

---

## Risk Assessment

### High Risk (Must Address)

1. **Insufficient training data** (CURRENT BLOCKER)
   - Current: 63 images
   - Target: 83 images
   - **Mitigation:** Complete Phase 1 data collection immediately

2. **Weak vein visibility at 22-25cm**
   - Edge density: 0.0053-0.0078 (target: >0.015)
   - **Mitigation:** Use adjusted capture settings (lower exposure, higher contrast)

3. **Training time uncertainty**
   - Estimated 4-6 hours, but could be longer
   - **Mitigation:** Monitor training progress, use early stopping

### Medium Risk

4. **OOD capture for VP-5**
   - Requires additional session at 18cm and 38cm
   - **Mitigation:** Can be done in parallel with Wave 4-5 tasks

5. **Threshold calibration sensitivity**
   - Grid search may not find optimal point
   - **Mitigation:** Manual refinement if needed, document tradeoffs

### Low Risk

6. **Module integration issues**
   - Well-isolated changes with unit tests
   - **Mitigation:** VP-1 and VP-2 catch regressions early

---

## Success Criteria (Updated Based on Dataset Analysis)

### Realistic Targets with 83 Images:

| Metric | Original Target | Revised Target | Rationale |
|--------|----------------|----------------|-----------|
| Intra-distance accuracy | ≥95% | ≥95% | Achievable with 20 samples at 27cm |
| Cross-distance accuracy (±5cm) | ≥95% | ≥90% | Limited by sample size (12-18 per distance) |
| Boundary accuracy (22cm, 32cm) | ≥95% | ≥85% | Boundary cases harder with limited data |
| Cross-hand false-accept | 0% | 0% | Non-negotiable (security requirement) |
| OOD rejection (18cm, 38cm) | ≥90% | ≥90% | Achievable with OOD detector |

**Note:** If cross-distance accuracy falls below 90%, consider:
- Capturing additional 5 samples per distance (total 108 images)
- More aggressive scale augmentation (0.75-1.35 range)
- Longer training (150 epochs instead of 100)

---

## Conclusion

**IMMEDIATE ACTION REQUIRED:** Complete Phase 1 data collection before proceeding with training. The current 63 images are insufficient for robust cross-distance recognition.

**Critical path:** Data collection (30 min) → Wave 1 tasks (3 hours) → Wave 2 tasks (2 hours) → Training (4-6 hours) → Deployment (1 hour) → Verification (5 hours)

**Total time to completion:** 15-18 hours (can be reduced to 12-15 hours with parallelization)

**Next immediate step:** Run the capture session following `capture_guide_next_session.md` to collect +20 images.

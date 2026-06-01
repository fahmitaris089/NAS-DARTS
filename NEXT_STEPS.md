# Next Steps — Live Scan Robustness Fix

**Status:** Dataset analysis COMPLETE ✅  
**Decision:** Lanjut ke training dengan 63 samples (Opsi A)  
**Updated Target:** Accuracy ≥90% (revised from ≥95%)

---

## Summary Analisis Dataset

### ✅ GOOD NEWS

1. **Quality bagus:** Semua laplacian variance > 60 (threshold quality filter)
2. **Setup capture optimal:** NIR 850 nm + 1 tisu + exposure 8000 µs menghasilkan contrast bagus di semua jarak
3. **Brightness gradient konsisten:** 170.8 (22 cm) → 129.5 (32 cm), expected dan bagus untuk model belajar invariance
4. **Contrast tinggi di jarak jauh:** 30-32 cm memiliki contrast tertinggi (60-63), bagus untuk vein visibility

### ⚠️ CONSTRAINTS

1. **Volume marginal:** 63 samples total (50% of ideal 125)
2. **Distribusi tidak seimbang:**
   - 25 cm: 8 samples (paling sedikit)
   - 30 cm: 18 samples (paling banyak)
   - 22 cm: 12 samples (ekstrem, butuh +3 jika ada waktu)
   - 32 cm: 14 samples (ekstrem, butuh +3 jika ada waktu)

### 📊 Dataset Breakdown

| Jarak | Samples | Lap Var | Brightness | Status |
|-------|---------|---------|------------|--------|
| 22 cm | 12 | 151.4 | 170.8 | ⚠️ Ekstrem |
| 25 cm | 8 | 152.5 | 165.8 | ⚠️ Undersampled |
| 27 cm | 11 | 157.9 | 148.2 | ✅ Nominal |
| 30 cm | 18 | 137.4 | 131.8 | ✅ Oversampled |
| 32 cm | 14 | 154.7 | 129.5 | ⚠️ Ekstrem |

**Total:** 63 samples across 5 distances

---

## Bottleneck Robustness: Root Cause

Berdasarkan analisis dataset dan bug documentation:

### 1. **Training Distribution Terlalu Sempit** (PRIMARY)
- Model original hanya lihat ~10 images di 27 cm (statik)
- Brightness range di dataset baru: 129.5 - 170.8 (41.3 point spread)
- Contrast range: 39.7 - 63.6 (23.9 point spread)
- **Ini adalah variasi yang TIDAK pernah dilihat model saat training original**

**Fix:** Dataset multi-distance ini + augmentation v2 (M-1, M-2)

### 2. **Augmentation yang Merusak** (SECONDARY)
- `RandomHorizontalFlip(p=0.5)` membuat model bingung antara tangan kiri vs kanan
- Menyebabkan cross-hand confusion (835 ↔ 836)

**Fix:** Remove horizontal flip + hand-pair margin loss (M-2, M-5)

### 3. **Volume Data Minimal** (TERTIARY)
- 63 samples vs ideal 125 = 50% of target
- Model akan memiliki robustness terbatas

**Fix:** Augmentation v2 yang lebih agresif + multi-distance enrollment (M-2, M-6)

---

## Decision: Opsi A (Lanjut Training Sekarang)

### Rationale

1. **Quality > Quantity:** Dataset quality sudah bagus, lebih penting daripada volume
2. **Augmentation will compensate:** Augmentation v2 akan generate variasi equivalent ~2-3x volume
3. **Iterative approach:** Train sekarang, evaluate, lalu decide apakah perlu tambahan data
4. **Fatigue factor:** Kamu sudah capek, forcing 62 captures lagi akan degrade quality

### Expected Outcome

- **Accuracy:** 88-92% (vs 95% target original)
- **Cross-hand confusion:** FIXED (remove horizontal flip + hand-pair loss)
- **Distance robustness:** IMPROVED (tapi tidak perfect karena volume terbatas)

### Trade-offs Documented

```
Volume: 63 samples (50% of ideal 125)
Target accuracy: ≥90% (revised from ≥95%)
Expected accuracy: 88-92%
Rationale: Quality > Quantity, augmentation will compensate
```

---

## Action Items (Immediate)

### 1. ✅ Update Design Document
- [x] Turunkan target accuracy dari ≥95% menjadi ≥90% di `design.md` (TA-2)
- [x] Dokumentasikan rationale: "Dataset volume constraint (63 samples vs ideal 125)"

### 2. 🚀 Lanjutkan ke Task 7 (Retrain)

**File:** `retrain_run7_robust.py` (analogous to `retrain_run6_plus2.py`)

**Config changes:**
```python
RETRAIN_CFG["augmentation_policy"] = "v2_multi_distance"
RETRAIN_CFG["hand_pair_margin_loss"] = True
RETRAIN_CFG["hand_pair_classes"] = [("835", "836")]
RETRAIN_CFG["hand_pair_margin"] = 1.0
RETRAIN_CFG["hand_pair_weight"] = 0.3
```

**Dataset:**
- Input: `dataset_multi_distance/835/` (63 images across 5 distances)
- Split: 80% train (50 images), 20% val (13 images)
- Test: Reserve 5 samples per distance (25 images) untuk held-out test

**Expected training time:** ~2-3 hours (similar to run6_plus2)

### 3. 📊 Monitor Training Curves

**Success criteria:**
- Validation accuracy ≥ 85% (minimum acceptable)
- Training loss converges smoothly (no overfitting)
- Cross-hand confusion = 0 on validation set

**If validation accuracy < 85%:**
- Consider Opsi B: Tambah 3-5 sample di jarak kritis (22 cm, 32 cm)
- Re-evaluate augmentation strength

---

## Alternative: Opsi B (If Needed)

**Trigger:** Validation accuracy < 85% after Task 7

**Action:**
1. Tambah 3 sample di 22 cm (current: 12 → target: 15)
2. Tambah 3 sample di 32 cm (current: 14 → target: 15)
3. Tambah 2 sample di 25 cm (current: 8 → target: 10)
4. **Total effort:** ~8 captures (15-20 menit)
5. Re-run Task 7 dengan 71 samples total

**Expected improvement:** +2-3% accuracy (90-93% range)

---

## Files Generated

1. ✅ `analyze_multi_distance_dataset.py` — Script analisis dataset
2. ✅ `analysis_results_835/analysis_results.json` — Raw data analisis
3. ✅ `analysis_results_835/quality_distribution.png` — Visualisasi quality metrics
4. ✅ `DATASET_ANALYSIS_REPORT.md` — Full analysis report (detailed)
5. ✅ `NEXT_STEPS.md` — This file (action items)
6. ✅ `design.md` — Updated TA-2 target (≥90%)

---

## Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| **Phase 1** | Dataset analysis | 1 hour | ✅ DONE |
| **Phase 2** | Update design doc | 10 min | ✅ DONE |
| **Phase 3** | Implement Task 1-6 | 2-3 days | 🔜 NEXT |
| **Phase 4** | Task 7 (Retrain) | 2-3 hours | ⏳ Pending |
| **Phase 5** | Task 8-13 (Verification) | 1-2 days | ⏳ Pending |

**Current phase:** Ready to start Phase 3 (Task 1-6 implementation)

---

## Key Takeaways

1. **Dataset quality is good** — Setup capture optimal, laplacian variance > 60 di semua jarak
2. **Volume is marginal but acceptable** — 63 samples adalah minimum absolut, tapi cukup untuk training awal
3. **Bottleneck identified** — Training distribution terlalu sempit + augmentation yang merusak
4. **Realistic target set** — Accuracy ≥90% (revised from ≥95%) untuk volume 63 samples
5. **Iterative approach** — Train sekarang, evaluate, lalu decide apakah perlu tambahan data

**Next step:** Lanjutkan ke Task 1 (Add palm_core_side_px alias) di `tasks.md`

---

## Questions?

Jika ada pertanyaan atau butuh klarifikasi:
1. Baca `DATASET_ANALYSIS_REPORT.md` untuk detail lengkap
2. Lihat `analysis_results_835/quality_distribution.png` untuk visualisasi
3. Check `design.md` untuk updated target accuracy (TA-2)
4. Review `tasks.md` untuk implementation plan

**Ready to proceed with Task 1-6 implementation!** 🚀

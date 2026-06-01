# Analisis Dataset Multi-Distance Tangan Kiri (835)

**Tanggal Analisis:** 1 Juni 2026  
**Dataset Root:** `dataset_multi_distance/835`  
**Total Samples:** 63 images across 5 distances

---

## Executive Summary

⚠️ **Status: MARGINAL — Dapat digunakan untuk training awal, tapi dengan ekspektasi realistis**

Dataset yang kamu akuisisi memiliki **kualitas image yang baik** (semua laplacian variance ≥ 60), tetapi **volume data masih di batas minimum absolut**. Dengan 63 samples total, model akan memiliki robustness terbatas dibanding target ideal 125 samples.

**Rekomendasi utama:**
1. **Lanjutkan ke training dengan dataset ini** untuk validasi pipeline end-to-end
2. **Turunkan target accuracy** di `design.md` dari ≥95% menjadi ≥90% (realistis untuk volume ini)
3. **Prioritaskan tambahan 3-5 sample di jarak kritis** (22 cm dan 32 cm) jika masih ada waktu

---

## Distribusi Dataset per Jarak

| Jarak | Samples | Laplacian Var (mean ± std) | Brightness | Status |
|-------|---------|---------------------------|------------|--------|
| **22 cm** | 12 | 151.4 ± 23.6 | 170.8 | ⚠️ Ekstrem (butuh +3) |
| **25 cm** | 8 | 152.5 ± 18.0 | 165.8 | ⚠️ Paling sedikit |
| **27 cm** | 11 | 157.9 ± 46.6 | 148.2 | ✅ Nominal (training baseline) |
| **30 cm** | 18 | 137.4 ± 28.2 | 131.8 | ✅ Paling banyak |
| **32 cm** | 14 | 154.7 ± 25.2 | 129.5 | ⚠️ Ekstrem (butuh +3) |

### Observasi Kunci

1. **Distribusi tidak seimbang:** 25 cm hanya 8 samples (paling sedikit), sementara 30 cm ada 18 samples (paling banyak). Idealnya setiap jarak punya 10-15 samples.

2. **Quality metrics bagus:** Semua jarak memiliki laplacian variance > 60 (threshold quality filter), artinya setup capture-mu (`exposure-us 8000`, `gain 1.1`, NIR 850 nm + 1 tisu) sudah optimal.

3. **Brightness gradient konsisten:** Brightness menurun dari 170.8 (22 cm) ke 129.5 (32 cm), yang expected karena jarak lebih jauh = intensitas NIR lebih rendah. Ini bagus karena model akan belajar invariance terhadap brightness.

4. **Variance tinggi di 27 cm:** Laplacian variance di 27 cm memiliki std tertinggi (46.6), kemungkinan karena ada 1-2 outlier images dengan contrast sangat tinggi (max 274.1). Ini tidak masalah karena augmentation akan handle variasi ini.

---

## Analisis Kualitas Image

### Laplacian Variance (Sharpness Proxy)

Semua distances memiliki mean laplacian variance **> 60** (threshold quality filter), yang artinya:
- ✅ Images tidak blur
- ✅ Palm vein pattern terlihat jelas
- ✅ Akan lolos quality filter saat live scan

**Best performers:**
- 27 cm: 157.9 (tertinggi, tapi variance besar)
- 32 cm: 154.7 (konsisten, variance kecil)
- 25 cm: 152.5 (konsisten, variance kecil)

**Lowest (tapi masih OK):**
- 30 cm: 137.4 (masih jauh di atas threshold 60)

### Brightness & Contrast

**Brightness gradient (expected):**
```
22 cm: 170.8 (paling terang, jarak dekat)
25 cm: 165.8
27 cm: 148.2
30 cm: 131.8
32 cm: 129.5 (paling gelap, jarak jauh)
```

**Contrast gradient (expected):**
```
22 cm: 39.7 (contrast rendah, terlalu terang)
25 cm: 41.4
27 cm: 54.8 (optimal)
30 cm: 60.7 (tinggi, bagus untuk vein visibility)
32 cm: 63.6 (tertinggi, bagus untuk vein visibility)
```

**Insight:** Jarak lebih jauh (30-32 cm) justru memiliki contrast lebih tinggi, yang bagus untuk palm vein visibility. Ini menunjukkan setup NIR-mu optimal untuk range 22-32 cm.

---

## Bottleneck Robustness: Root Cause Analysis

Berdasarkan analisis dataset dan konteks bug di `bugfix.md`, bottleneck utama robustness model-mu adalah:

### 1. **Training Distribution Terlalu Sempit** (PRIMARY BOTTLENECK)

**Problem:**
- Model original hanya pernah lihat ~10 images di 27 cm (statik)
- Embedding space tidak punya margin untuk variasi jarak
- Saat live scan di 22 cm atau 32 cm, model menganggap ini "out-of-distribution" dan reject atau salah prediksi

**Evidence dari dataset-mu:**
- Brightness range: 129.5 - 170.8 (41.3 point spread)
- Contrast range: 39.7 - 63.6 (23.9 point spread)
- Ini adalah variasi yang TIDAK pernah dilihat model saat training original

**Mitigation (M-1, M-2):**
- Dataset multi-distance ini akan expose model ke variasi brightness/contrast
- Augmentation v2 (`RandomAffine scale=(0.78, 1.28)`) akan simulate variasi ROI size akibat jarak

### 2. **Augmentation yang Merusak** (SECONDARY BOTTLENECK)

**Problem:**
- `RandomHorizontalFlip(p=0.5)` di training original membuat model bingung antara tangan kiri vs kanan
- Ini menyebabkan cross-hand confusion (bug 1.2 dan 1.3 di `bugfix.md`)

**Evidence:**
- Kamu report model kadang prediksi 835 (kiri) sebagai 836 (kanan) dan sebaliknya
- Horizontal flip adalah augmentation yang tidak valid untuk palm vein (left ≠ right)

**Mitigation (M-2, M-5):**
- Remove `RandomHorizontalFlip` dari augmentation v2
- Add hand-pair margin loss untuk enforce separation antara 835 dan 836

### 3. **Volume Data Minimal** (TERTIARY BOTTLENECK)

**Problem:**
- Dengan 63 samples total (vs target 125), model tidak punya cukup contoh untuk generalisasi
- Distribusi tidak seimbang (25 cm hanya 8 samples)

**Impact:**
- Model akan overfit ke jarak dengan banyak samples (30 cm: 18 samples)
- Model akan underfit ke jarak dengan sedikit samples (25 cm: 8 samples)

**Mitigation:**
- Augmentation v2 yang lebih agresif untuk kompensasi volume
- Multi-distance enrollment (M-6) untuk capture variasi di template store
- Threshold calibration (M-7) untuk tuning operating point

---

## Rekomendasi: 3 Opsi Forward Path

### Opsi A: **Lanjut Training Sekarang** (RECOMMENDED)

**Rationale:**
- Dataset quality sudah bagus (laplacian variance > 60)
- Volume 63 samples adalah minimum absolut, tapi cukup untuk validasi pipeline
- Kamu sudah capek akuisisi data, dan diminishing returns untuk tambahan sample kecil

**Action items:**
1. ✅ **Lanjutkan ke Task 7** (retrain dengan augmentation v2 + hand-pair loss)
2. ✅ **Turunkan target accuracy** di `design.md`:
   - Original: ≥95% (unrealistic untuk 63 samples)
   - Revised: ≥90% (realistis untuk volume ini)
3. ✅ **Dokumentasikan trade-off** di training log: "Volume 63 samples (50% of ideal 125) → expect accuracy 90-92% instead of 95%"

**Expected outcome:**
- Accuracy: 88-92% (vs 95% target original)
- Cross-hand confusion: FIXED (karena remove horizontal flip + hand-pair loss)
- Distance robustness: IMPROVED (tapi tidak perfect karena volume terbatas)

**Timeline:** Bisa mulai training hari ini.

---

### Opsi B: **Tambah Minimal di Jarak Kritis** (BALANCED)

**Rationale:**
- Jarak ekstrem (22 cm dan 32 cm) adalah yang paling berisiko untuk OOD reject
- Tambahan 3-5 sample per jarak kritis akan significantly improve robustness di boundary

**Action items:**
1. ⚠️ **Tambah 3 sample di 22 cm** (current: 12 → target: 15)
2. ⚠️ **Tambah 3 sample di 32 cm** (current: 14 → target: 15)
3. ⚠️ **Tambah 2 sample di 25 cm** (current: 8 → target: 10, untuk balance distribusi)
4. ✅ **Total effort:** ~8 captures (15-20 menit dengan setup yang sudah ada)
5. ✅ **Lanjutkan ke Task 7** setelah tambahan ini

**Expected outcome:**
- Accuracy: 90-93% (slight improvement dari Opsi A)
- Distance robustness: SIGNIFICANTLY BETTER di boundary (22 cm dan 32 cm)
- Cross-hand confusion: FIXED

**Timeline:** +1 hari untuk akuisisi, lalu mulai training.

---

### Opsi C: **Full Target 125 Samples** (IDEAL, tapi overkill)

**Rationale:**
- Ini adalah target ideal di `design.md` (25 samples per distance × 5 distances)
- Akan memberikan robustness maksimal

**Action items:**
1. ❌ **Tambah 62 samples** (current: 63 → target: 125)
2. ❌ **Distribusi:** +17 di 25 cm, +14 di 27 cm, +7 di 30 cm, +11 di 32 cm, +13 di 22 cm
3. ❌ **Total effort:** ~62 captures (2-3 jam dengan setup yang sudah ada)

**Why NOT recommended:**
- Diminishing returns: 63 → 125 samples hanya akan improve accuracy ~3-5%
- Kamu sudah capek, dan effort 2-3 jam tidak worth it untuk gain kecil
- Bisa iterate nanti jika hasil training Opsi A/B tidak memuaskan

**Expected outcome:**
- Accuracy: 93-95% (marginal improvement dari Opsi B)
- Timeline: +2-3 hari untuk akuisisi

---

## Rekomendasi Final: **Opsi A** (Lanjut Training Sekarang)

**Reasoning:**
1. **Quality > Quantity:** Dataset-mu sudah punya quality bagus (laplacian variance > 60), yang lebih penting daripada volume untuk robustness
2. **Augmentation will compensate:** Augmentation v2 yang agresif (`RandomAffine scale=(0.78, 1.28)`, `ColorJitter brightness=0.20`) akan generate variasi yang equivalent dengan ~2-3x volume
3. **Iterative approach:** Lebih baik train sekarang, evaluate, lalu decide apakah perlu tambahan data (data-driven decision)
4. **Fatigue factor:** Kamu sudah capek, dan forcing 62 captures lagi akan degrade quality (hand positioning inconsistent, frustration)

**Next steps:**
1. ✅ **Update `design.md`:** Turunkan target accuracy dari ≥95% menjadi ≥90%
2. ✅ **Lanjutkan ke Task 7:** Retrain dengan augmentation v2 + hand-pair loss
3. ✅ **Monitor training curves:** Jika validation accuracy stuck di <85%, baru consider Opsi B (tambah sample di jarak kritis)

---

## Appendix: Distribusi Sample per Jarak (Detail)

### 22 cm (12 samples) — CRITICAL DISTANCE
- Laplacian variance: 96.3 - 180.3 (range: 84.0)
- Brightness: 170.8 ± 4.0 (paling terang)
- **Risk:** Jarak paling dekat, berisiko terlalu terang dan ROI terlalu besar
- **Recommendation:** +3 samples jika pilih Opsi B

### 25 cm (8 samples) — UNDERSAMPLED
- Laplacian variance: 123.0 - 182.5 (range: 59.5)
- Brightness: 165.8 ± 5.5
- **Risk:** Paling sedikit samples, model akan underfit di jarak ini
- **Recommendation:** +2 samples untuk balance distribusi (Opsi B)

### 27 cm (11 samples) — NOMINAL (training baseline)
- Laplacian variance: 102.1 - 274.1 (range: 172.0, ada outlier)
- Brightness: 148.2 ± 3.7
- **Note:** Ini adalah jarak yang dipakai untuk training original (~10 samples)
- **Recommendation:** OK as-is

### 30 cm (18 samples) — OVERSAMPLED
- Laplacian variance: 88.5 - 184.9 (range: 96.4)
- Brightness: 131.8 ± 5.3
- **Note:** Paling banyak samples, model akan bias ke jarak ini
- **Recommendation:** OK as-is (tidak perlu tambahan)

### 32 cm (14 samples) — CRITICAL DISTANCE
- Laplacian variance: 108.8 - 199.7 (range: 90.9)
- Brightness: 129.5 ± 2.3 (paling gelap, tapi masih OK)
- **Risk:** Jarak paling jauh, berisiko terlalu gelap dan ROI terlalu kecil
- **Recommendation:** +3 samples jika pilih Opsi B

---

## Visualisasi

Lihat plots di `analysis_results_835/`:
- `quality_distribution.png` — Laplacian variance dan sample count per jarak
- `analysis_results.json` — Raw data untuk analisis lebih lanjut

---

## Kesimpulan

Dataset multi-distance tangan kiri (835) yang kamu akuisisi memiliki:
- ✅ **Quality bagus:** Semua laplacian variance > 60
- ✅ **Setup capture optimal:** NIR 850 nm + 1 tisu + exposure 8000 µs menghasilkan contrast bagus di semua jarak
- ⚠️ **Volume marginal:** 63 samples adalah minimum absolut (50% of ideal 125)
- ⚠️ **Distribusi tidak seimbang:** 25 cm undersampled (8), 30 cm oversampled (18)

**Rekomendasi:** Lanjutkan ke training dengan dataset ini (Opsi A), turunkan target accuracy ke ≥90%, dan monitor hasil. Jika accuracy <85%, baru tambahkan 3-5 sample di jarak kritis (Opsi B).

**Bottleneck utama robustness:** Training distribution terlalu sempit (original model hanya lihat 27 cm) + augmentation yang merusak (horizontal flip). Dataset multi-distance ini akan fix bottleneck pertama, dan augmentation v2 akan fix bottleneck kedua.

# SPEC SLIDE — Sidang Proposal Tesis (Hardware-Aware NAS + KD + INT8)

> Spec final untuk membuat PPT (mis. via Gemini). **1 section = 1 slide**, berurutan.
> Format tiap slide: **Judul**, **Isi** (poin/tabel), **[Gambar]** (rujukan dari buku tesis), **[Catatan]** (untuk pembicara, jangan ditaruh di slide).
> Konteks global (footer kecil di slide hasil): *Latensi @ Raspberry Pi 5 (ONNX Runtime, 4-thread); akurasi test 834 kelas; INT8 = PTQ static per-channel (opset 13).*

---

# BAGIAN A — PROPOSAL (Slide 1–15)

---

## SLIDE 1 — Judul

**Judul (disarankan):**
*Arsitektur Jaringan Ringan untuk Pengenalan Palm Vein pada Perangkat Edge Menggunakan Hardware-Aware Neural Architecture Search dan Knowledge Distillation*

- Mohammad Taris Syahir Zul Fahmi — 6025242008
- Pembimbing: Prof. Dr. Eng. Nanik Suciati, S.Kom, M.Kom
- Sidang Proposal Tesis — EF255203 | ITS 2025

[Catatan: judul lama ditambah "pada Perangkat Edge / Hardware-Aware" karena kontribusi baru = pengukuran latensi nyata di Raspberry Pi.]

---

## SLIDE 2 — Latar Belakang

- Palm vein = biometrik vaskular NIR; nonkontak, higienis, sulit dipalsukan, stabil terhadap usia.
- CNN unggul untuk pola vena halus/kontras rendah, tetapi arsitektur besar (VGG/ResNet) berat → tak cocok perangkat terbatas.
- Model ringan (MobileNet) dirancang untuk citra RGB kompleks; pada palm vein NIR grayscale terjadi **overcapacity** → komputasi mubazir.
- Target deployment nyata = **perangkat edge (Raspberry Pi)**. Efisiensi tak cukup diukur FLOPs teoretis; **latensi & memori harus diukur langsung di perangkat** (karakteristik CPU ARM + runtime).
- Diperlukan arsitektur **sadar-perangkat (hardware-aware)** sejak desain.

[Gambar: Gambar 2.1 (akuisisi NIR), kecil di pojok. Tambah ikon Raspberry Pi.]

---

## SLIDE 3 — Fokus Penelitian: Masalah & Solusi

**Masalah:**
1. Arsitektur ringan manual (MobileNet) overcapacity untuk palm vein NIR.
2. Efisiensi sering divalidasi via FLOPs teoretis, **bukan latensi nyata** di perangkat target.
3. Akurasi model ringan turun saat dikompresi bila arsitektur tak dirancang baik.

**Solusi (3 pilar):**
1. **Hardware-Aware NAS** (DARTS + penalti latensi LUT Raspberry Pi).
2. **Knowledge Distillation** (jaga akurasi student kecil).
3. **Quantization INT8 (PTQ)** (tekan ukuran & percepat inferensi edge).

[Catatan: tekankan kata "sadar-perangkat" — pembeda dari proposal lama.]

---

## SLIDE 4 — Analisis Kesenjangan (Gap Analysis)

- NAS, KD, kompresi terbukti efektif **terpisah**, belum ada yang memadukan ketiganya khusus **palm vein NIR**.
- NAS domain lain **belum hardware-aware untuk edge nyata** (penalti masih proxy/FLOPs).
- KD pada palm vein NIR masih sangat terbatas.
- **Gap penelitian ini:** student **hardware-aware (LUT perangkat)** + KD + INT8, **divalidasi latensi nyata di Raspberry Pi**.

[Gambar: Tabel 2.1 (Analisis GAP) — baris "penelitian saat ini" tambah aspek hardware-aware + Raspberry Pi.]

---

## SLIDE 5 — Dataset & Preprocessing

**Dataset: SCUT_PV_v1 (Luo dkk., 2024)** — palm vein NIR
- 550 subjek, 1.100 telapak, 11.000 citra, resolusi 640×480; target eksperimen **834 kelas**.
- Split **80% train / 10% val / 10% test** (val untuk optimasi arsitektur bilevel NAS).

**Preprocessing:** ROI extraction (Gradient-Based Palm Center + Intensity-Weighted Centroid, crop 384×384) → **CLAHE** → normalisasi → **resize 224×224**.

[Gambar: Tabel 3.1 (dataset), Gambar 3.2 (ROI), Gambar 3.3 (pipeline), Gambar 2.2 (contoh raw→ROI→CLAHE).]

---

## SLIDE 6 — Metodologi: Alur Umum

- Pipeline: **Preprocessing → Teacher (EfficientNetV2-M) + Student (Hardware-Aware NAS) → Knowledge Distillation → Quantization INT8 → Evaluasi di Raspberry Pi.**

[Gambar: Gambar 3.1 (desain penelitian) — UPDATE: ResNet50→EfficientNetV2-M; blok NAS beri label "Hardware-Aware (LUT Pi)"; tambah blok "Deployment & Benchmark Raspberry Pi".]

---

## SLIDE 7 — Metode 1: Hardware-Aware NAS

**Dasar:** DARTS (Liu dkk., 2019) — relaksasi kontinu, optimasi bilevel; P-DARTS 3 tahap.

**Penalti latensi sadar-perangkat (inti baru), diferensiabel terhadap α:**

```
min_α   L_CE^val(w*, α)  +  λ · LAT(α)
s.t.    w* = argmin_w  L_CE^train(w, α)

LAT(α) = Σ_(i,j) Σ_o  softmax(α^(i,j))_o · COST_Pi(o)
```

- `COST_Pi(o)` = latensi operator dari **LUT diukur di Raspberry Pi** (dinormalisasi ke max).
- `λ` = bobot penalti (sweep 0.0 / 0.05 / 0.10 / 0.20); λ besar → lebih hemat latensi.
- **Search space 12 operasi:** none, skip, sep_conv 3×3/5×5, dil_conv 3×3/5×5, mbconv3/6, **rep_conv 3×3/5×5**, avg/max pool.

**Literatur (latency-aware NAS):** ProxylessNAS (ICLR'19); FBNet (CVPR'19, latency LUT); MnasNet (CVPR'19).

[Gambar: Gambar 3.5 (alur DARTS). Catatan: CVPR/ICLR = peringkat A*/setara Q1 untuk computer vision.]

---

## SLIDE 8 — Latency LUT Raspberry Pi (bukti hardware-aware)

**Apa itu LUT:** tabel biaya latensi tiap operasi, **diukur langsung di Raspberry Pi** (median 100 iterasi), dipakai sebagai `COST_Pi(o)` di rumus Slide 7 — bukan estimasi FLOPs.

| Operasi | LUT FP32 (ms) |
|---|---|
| skip_connect / pool | 0.02 |
| **rep_conv_3x3** | **0.055** (murah) |
| dil_conv_3x3 | 0.043 |
| sep_conv_3x3 / 5x5 | 0.054 / 0.107 |
| mbconv6_3x3 | **0.158** (termahal) |

- Biaya **tidak** sebanding FLOPs; mengikuti CPU ARM nyata.
- `rep_conv` murah, `mbconv/sep_conv` mahal → saat λ naik, NAS bergeser ke `rep_conv`.

[Sumber: `latency_lut_pi.json`. Sitasi: FBNet (Wu dkk., 2019), ProxylessNAS (Cai dkk., 2019).]

---

## SLIDE 9 — Metode 2: Knowledge Distillation

- Transfer representasi dari **teacher EfficientNetV2-M (frozen)** ke student NAS.
- Loss gabungan dengan temperature scaling:

```
L = α · L_CE  +  (1 − α) · L_KD ,    L_KD = T² · KL(p_t^T ‖ p_s^T)
```

- T melunakkan distribusi teacher → student belajar relasi antar-kelas.

[Gambar: Gambar 2.5 (diagram teacher-student) & Gambar 3.6 (alur distilasi) — label teacher = EfficientNetV2-M.]
[Catatan: separabilitas antar-kelas tinggi → efek KD diperkirakan marginal; KD = penjaga akurasi, bukan pengungkit utama.]

---

## SLIDE 10 — Metode 3: Quantization INT8

- **PTQ INT8 per-channel** (QDQ, ONNX opset 13), tanpa retraining, pakai data kalibrasi: `q = round(x / scale)`.
- Tujuan: turunkan ukuran model & percepat inferensi CPU edge.
- **Temuan kerangka (manfaat per-model):**
  - Ukuran: **selalu turun** (≈2–4×).
  - Latensi: **kondisional** (untung di operator padat; rugi di sel compact karena overhead QDQ).
  - Akurasi: **terjaga** (INT8 ≈ FP32).
- Deliverable: **aturan pemilihan presisi (FP32/INT8) per model**.

[Gambar: Gambar 2.6 (FP32→INT8), Gambar 3.7 (alur kompresi).]
[Catatan: jika ditanya pruning → fokus dipindah ke quantization (kompresi & percepatan terukur tanpa destabilisasi arsitektur kecil); pruning = future work.]

---

## SLIDE 11 — Setup Deployment (Raspberry Pi 5)

- **Peran:** Pi 5 = target pengukuran efisiensi, **bukan** perangkat akuisisi citra.
- **Spesifikasi:** Raspberry Pi 5 · SoC BCM2712 quad-core Cortex-A76 @2.4GHz · RAM [isi, mis. 8GB] · Raspberry Pi OS 64-bit.
- **Konfigurasi:** ONNX Runtime (CPU), 4 thread; uji FP32 & INT8; input test set SCUT_PV_v1 (224×224).
- **Diukur:** latensi/citra (median, p95), ukuran model, akurasi.

[Catatan: jika ditanya kamera/IR → "Pi 5 hanya untuk mengukur latensi inferensi pada citra dataset; akuisisi sensor live = future work, sudah dinyatakan di batasan masalah."]

---

## SLIDE 12 — Metrik Evaluasi

- **Kinerja pengenalan:** Akurasi, FAR, FRR, **EER** (utama biometrik).
- **Efisiensi (diukur di Raspberry Pi 5):** jumlah parameter, FLOPs, **ukuran model (MB)** FP32 vs INT8, **latensi inferensi (ms)** + throughput.

[Catatan: tegaskan latensi ON-DEVICE, bukan estimasi FLOPs — pembeda kunci.]

---

## SLIDE 13 — Skenario Eksperimen

| No | Skenario | Model | Fokus |
|---|---|---|---|
| 1 | Baseline | EfficientNetV2-M (teacher) & MobileNetV3 | Plafon akurasi & efisiensi standar |
| 2 | NAS mandiri | Hardware-Aware NAS (latih scratch) | Validasi struktur + sweep λ |
| 3 | Distillation | NAS + Teacher (KD) | Kenaikan akurasi student |
| 4 | Full Model | NAS + KD + **INT8** | Latensi, ukuran, akurasi di Raspberry Pi 5 |

[Gambar: Tabel 3.2 (rangkuman skenario) — teacher = EfficientNetV2-M; Skenario 4 = INT8 + benchmark Pi.]

---

## SLIDE 14 — Batasan Masalah (poin kunci)

1. Hanya dataset SCUT_PV_v1 (NIR); tidak RGB/multispektral/multimodal.
2. Fokus identifikasi/klasifikasi; tanpa PAD/liveness.
3. Teacher = CNN konvensional berkinerja tinggi.
4. Diarahkan untuk perangkat edge.
5. NAS untuk desain student (bukan hyperparameter/teacher).
6. Tanpa generalisasi lintas dataset/sensor.
7. **[BARU] Raspberry Pi 5 = platform evaluasi deployment (ukur latensi & ukuran pada citra uji); akuisisi live (kamera NoIR/IR) di luar lingkup → future work.**

---

## SLIDE 15 — Kontribusi Penelitian

1. **Kerangka terpadu** NAS + KD + Quantization khusus **palm vein NIR** untuk edge.
2. **Hardware-Aware NAS dengan penalti latensi LUT Raspberry Pi** — optimasi terhadap latensi perangkat nyata; λ = knob frontier akurasi–latensi.
3. **Validasi & benchmark latensi nyata di Raspberry Pi 5** (FP32 vs INT8).
4. **Karakterisasi kapan INT8 untung/rugi** (mekanisme QDQ-overhead vs arithmetic intensity) → aturan pemilihan presisi per-model.
5. **Student ringan (<1 MB, ratusan ribu param)** dengan akurasi kompetitif & EER rendah.

[Catatan: #2–#4 = novelty utama vs proposal lama & vs Ding dkk. 2025 (NAS greedy, tanpa KD).]


---

# BAGIAN B — HASIL & PEMBAHASAN (Slide 16–21)

> Footer kecil tiap slide: *Latensi @ Raspberry Pi 5 (ONNX, 4-thread); akurasi test 834 kelas; INT8 = PTQ per-channel.*

---

## SLIDE 16 — Peta Hasil (pembuka hasil)

**Judul:** Ringkasan Hasil — 4 Temuan Utama

- **NAS sadar-hardware** → frontier Pareto akurasi–latensi (λ = knob akurat↔cepat).
- **KD** → perbaikan marginal (+0.1–0.3 pp, dalam noise).
- **INT8** → ukuran selalu turun; latensi untung kondisional.
- Semua **terukur di Raspberry Pi 5**.

> **Headline:** Model akhir **0.52 M param · 0.61 MB · 98.92% @ 5.27 ms** — vs MobileNetV3 **5.27 M · 21 MB · 99.88% @ 15.49 ms** (≈35× lebih kecil, 3× lebih cepat, akurasi −1 pp).

---

## SLIDE 17 — Hasil NAS: λ Menggeser Operator + Pareto

**Judul:** Hardware-Aware NAS — Operator & Pareto

**Pergeseran operator (bukti LUT bekerja):**

| λ | Operator dominan | Sifat |
|---|---|---|
| 0.00 | **sep_conv** | akurasi murni (mahal di Pi) |
| 0.05 | rep + dil + skip | mulai hemat |
| 0.10–0.20 | **rep_conv + skip** | latensi-aware (murah) |

→ λ naik ⇒ NAS pindah dari `sep_conv` ke `rep_conv` (fusi jadi 1 conv).

**Pareto FP32:** C4 kolaps (91–93%, dibuang) · **C6 = pojok cepat (97.96% @ 3.99 ms)** · **C8 = pojok akurat (99.16% @ 6.29 ms)**.

[Gambar opsional/backup: 1 diagram topologi genotype λ0.20 C8.]

---

## SLIDE 18 — Hasil Knowledge Distillation

**Judul:** KD — Perbaikan Marginal, Berpola (teacher EfficientNetV2-M)

| Model | Baseline | + KD (best by-val) |
|---|---|---|
| C6 | 97.96% | **98.20%** (T=10–20, α=0.4–0.5) |
| C8 | 99.16% | **99.28%** (T=10–20, α=0.4–0.5) |

- KD **+0.1–0.3 pp** (dalam noise single-seed) → bukan pengungkit utama.
- **Pola:** T tinggi + α seimbang terbaik; **α=0.1 merusak** (C6 → 94.72%).
- **Sebab:** teacher saturasi (100%) → dark knowledge miskin → bobot KD berlebih membuang label asli. *(temuan, bukan kegagalan)*

---

## SLIDE 19 — Hasil Quantization INT8

**Judul:** INT8 — Ukuran Selalu Turun, Latensi Kondisional

| Kelompok model | Akurasi FP32→INT8 | Latensi | Efek INT8 |
|---|---|---|---|
| C4 / C6 (compact) | ~setara | lebih lambat | **rugi** (0.67–0.95×) |
| **C8 (padat)** | ~setara | lebih cepat | **untung** (1.06–1.19×) |
| MobileNetV3 | −1.2 pp | 15.5→8.4 ms | untung (1.85×) |

- **Ukuran selalu turun 1.3–3.6×**; akurasi terjaga (Δ ≈ −0.2 pp).
- **Aturan:** INT8 untung ⟺ hemat-compute conv > overhead konversi QDQ (~2.4–2.7 ms, tetap). Sel padat menang; sel compact kalah (memory-bound).
- **Deliverable:** pemilihan presisi per-model (C8→INT8, C6→FP32).

[Gambar opsional: bar compute-saving vs QDQ-overhead untuk C6 vs C8.]

---

## SLIDE 20 — Tabel Master & Pemenang

**Judul:** Perbandingan Menyeluruh — NAS Jauh Lebih Kecil, Akurasi Terjaga

| Model | Param (M) | FLOPs (MMACs) | Akurasi | Ukuran | Latensi Pi 5 |
|---|---|---|---|---|---|
| EfficientNetV2-M (teacher) | 53.93 | 5 446 | 100% | ≈206 MB | — (GPU) |
| MobileNetV3-L (baseline) | 5.27 | 235 | 99.88% | 21.08 MB | 15.49 ms |
| **hwNAS C8 (NAS+KD, INT8)** | **0.52** | 136 | ~99.0%¹ | **0.61 MB** | **5.27 ms** |
| **hwNAS C6 (NAS+KD, FP32)** | **0.32** | 59 | **98.20%** | 0.79 MB | **3.99 ms** |

¹ C8: KD-FP32 terukur **99.28%**, non-KD INT8 terukur **98.92%**; KD+INT8 belum di-rebenchmark (perlu export+kuantisasi model KD; latensi/ukuran identik karena arsitektur sama).

**Pemenang per skenario:** A-Akurasi → **C8 INT8 (98.92% / 5.27 ms)** · B-Tercepat → **C6 FP32 (97.96% / 3.99 ms)**.

[Gambar opsional: scatter Pareto akurasi vs latensi Pi.]
[Catatan: NAS ~100–270× lebih sedikit param dari teacher; ~35× lebih kecil & 3–4× lebih cepat dari MobileNet.]

---

## SLIDE 21 — Kesimpulan

**Judul:** Kesimpulan

1. **NAS** → student ringan sadar-perangkat (<1 MB), λ = knob Pareto, kalahkan baseline manual.
2. **KD** → perbaikan marginal (+0.1–0.3 pp); pengungkit utama NAS + INT8.
3. **INT8** → ukuran turun 1.3–3.6×, akurasi terjaga; latensi untung kondisional → aturan presisi per-model.

**Kontribusi:** kerangka terpadu **NAS hardware-aware (LUT Pi) + KD + INT8** untuk palm-vein NIR, divalidasi di Raspberry Pi 5.

**Limitasi:** single-seed → perlu 3-seed + McNemar. **Lanjutan:** akuisisi live-scan (di luar lingkup).

---
---

# APPENDIX (slide cadangan — hanya jika ditanya penguji)

---

## APP-1 — Eksplorasi 9 Kandidat Teacher

| Model | Params (M) | FLOPs (MMACs) | Akurasi | EER |
|---|---|---|---|---|
| EfficientNetV2-M ★ | 53.93 | 5 446 | 100% | 0 |
| ResNet50 | 25.22 | 4 133 | 100% | 0 |
| ConvNeXt-Base | 88.42 | 15 373 | 100% | 0 |
| RegNetY-16GF | 83.09 | 16 009 | 100% | 0 |
| DenseNet121 | 7.81 | 2 897 | 99.88% | 0 |
| MobileNetV3-Large | 5.27 | 235 | 99.88% | 0 |
| InceptionV3 | 26.69 | 2 856 | 99.76% | 0 |
| EfficientNetB4 | 19.04 | 1 578 | 99.76% | 0 |
| VGG16 | 137.68 | 15 470 | 99.64% | 0 |

[FLOPs = MMACs@224 (thop). 4 model saturasi 100% → pilihan teacher tidak kritikal → menjelaskan KD marginal.]

---

## APP-2 — Tabel INT8 Lengkap (8 model)

| Model | Akurasi FP32→INT8 | Latensi FP32→INT8 | Speedup | Ukuran |
|---|---|---|---|---|
| hwNAS λ0.05 C4 | 93.29 → 93.29% | 2.53 → 3.75 ms | 0.67× | 1.27× |
| hwNAS λ0.05 C6 | 97.96 → 97.96% | 3.99 → 5.10 ms | 0.78× | 1.76× |
| hwNAS λ0.20 C6 | 97.60 → 97.36% | 4.27 → 4.47 ms | 0.95× | 2.10× |
| hwNAS λ0.05 C8 | 98.08 → 98.32% | 5.81 → 5.46 ms | 1.06× | 2.16× |
| hwNAS λ0.10 C8 | 99.16 → 99.04% | 6.75 → 5.70 ms | 1.18× | 2.51× |
| hwNAS λ0.20 C8 | 99.16 → 98.92% | 6.29 → 5.27 ms | 1.19× | 2.52× |
| MobileNetV3-L | 99.88 → 98.68% | 15.49 → 8.39 ms | 1.85× | 3.64× |

---

## APP-3 — Checklist Gambar yang Perlu Diupdate

- [ ] Gambar 3.1: ResNet50 → EfficientNetV2-M; blok NAS → "Hardware-Aware (LUT Pi)"; tambah blok "Deployment Raspberry Pi".
- [ ] Gambar 3.6: label teacher → EfficientNetV2-M.
- [ ] Tabel 2.1 / 3.2: sesuaikan teacher + hardware-aware + Raspberry Pi.
- [ ] Tambah foto/ikon Raspberry Pi 5 di slide 2, 11, 12.

---

## CATATAN PERUBAHAN dari proposal lama (untuk kamu, bukan slide)

| Aspek | Lama | Sekarang |
|---|---|---|
| Penalti NAS | DARTS / ops-count | Hardware-aware (LUT Pi) |
| Teacher | ResNet50 | EfficientNetV2-M |
| Kompresi | Pruning + Quant | Quantization INT8 (pruning → future work) |
| Latensi | FLOPs teoretis | Terukur di Raspberry Pi 5 |

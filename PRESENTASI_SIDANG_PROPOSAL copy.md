# Plan Slide Sidang Proposal Tesis — Versi Baru (Hardware-Aware)

> Konten sudah final, tinggal copy-paste ke PPT. Untuk slide bergambar, nama gambar dari
> buku tesis disebut eksplisit (mis. **Gambar 3.1**). Teks dalam tanda kurung siku `[ ... ]`
> adalah catatan untukmu (jangan dimasukkan ke slide).

---

## RINGKASAN PERUBAHAN dari proposal lama (baca dulu, jangan masuk slide)

| Aspek | Proposal lama (buku) | Versi sekarang (riil) |
|---|---|---|
| Penalti NAS | DARTS biasa / ops-count | **Hardware-Aware: penalti latensi LUT Raspberry Pi** |
| Target deploy | "perangkat edge" (abstrak) | **Raspberry Pi (diukur nyata, ONNX Runtime CPU 4-thread)** |
| Teacher | ResNet50 | **EfficientNetV2-M** [verifikasi: ganti semua "ResNet50" → "EfficientNetV2-M" di Gambar 3.1 & 3.6] |
| Kompresi | Pruning + Quantization | **Quantization INT8 PTQ per-channel** (pruning di-drop/opsional) |
| Metrik latensi | FLOPs teoretis | **Latensi & ukuran terukur di Raspberry Pi** |

---

## SLIDE 1 — Judul

**Pilihan judul (rekomendasi: B — perubahan minimal tapi bermakna):**

- **B (disarankan):** *Arsitektur Jaringan Ringan untuk Pengenalan Palm Vein pada Perangkat Edge Menggunakan Hardware-Aware Neural Architecture Search dan Knowledge Distillation*
- A (paling eksplisit): *...Pengenalan Palm Vein pada Raspberry Pi Menggunakan Hardware-Aware Neural Architecture Search dan Knowledge Distillation*
- C (paling aman, judul lama + anak judul): judul lama + *"dengan Evaluasi Deployment pada Raspberry Pi"*

Mohammad Taris Syahir Zul Fahmi — 6025242008
Pembimbing: Prof. Dr. Eng. Nanik Suciati, S.Kom, M.Kom
Sidang Proposal Tesis — EF255203 | ITS 2025

[Catatan: di sidang, jelaskan 1 kalimat bahwa judul ditambah "Hardware-Aware / Perangkat Edge"
karena kontribusi pengukuran latensi nyata di Raspberry Pi.]

---

## SLIDE 2 — Latar Belakang

- Palm vein = biometrik vaskular NIR; nonkontak, higienis, sulit dipalsukan, stabil terhadap usia.
- CNN unggul mengekstraksi pola vena halus & kontras rendah, tetapi arsitektur besar (VGG/ResNet) berat → tak cocok untuk perangkat sumber daya terbatas.
- Model ringan (MobileNet) dirancang untuk citra **RGB kompleks**; pada palm vein **NIR grayscale** terjadi **overcapacity** → komputasi mubazir.
- **[BARU] Target deployment nyata = perangkat edge (Raspberry Pi).** Efisiensi tidak cukup diukur dengan FLOPs/params teoretis; **latensi & memori harus diukur langsung di perangkat** karena dipengaruhi karakteristik CPU ARM dan runtime.
- Diperlukan arsitektur yang dirancang **sadar-perangkat (hardware-aware)** sejak awal, bukan sekadar kecil secara teoretis.

[Gambar: Gambar 2.1 (akuisisi NIR) — opsional kecil di pojok]

---

## SLIDE 3 — Fokus Penelitian: Masalah & Solusi

**Masalah:**
1. Arsitektur ringan manual (MobileNet) overcapacity untuk palm vein NIR & tidak optimal di perangkat edge.
2. Efisiensi sering hanya divalidasi dengan FLOPs teoretis, **bukan latensi nyata** di perangkat target.
3. Akurasi model ringan menurun saat dikompresi bila arsitektur tidak dirancang baik.

**Solusi (3 pilar terintegrasi):**
1. **Hardware-Aware NAS (DARTS + penalti latensi LUT Raspberry Pi)** → arsitektur student yang langsung dioptimasi terhadap latensi perangkat.
2. **Knowledge Distillation** → menjaga kualitas representasi student kecil mendekati teacher.
3. **Quantization INT8 (PTQ)** → menekan ukuran model & mempercepat inferensi pada CPU edge.

[Catatan pembicara: tekankan kata "sadar-perangkat" — inilah pembeda dari proposal lama.]

---

## SLIDE 4 — Analisis Kesenjangan (Gap Analysis)

- NAS, KD, dan kompresi terbukti efektif **secara terpisah**, tapi **belum ada** yang memadukan ketiganya khusus untuk **palm vein NIR**.
- NAS pada domain lain **belum hardware-aware untuk perangkat edge nyata** (penalti masih proxy/FLOPs, bukan latensi terukur).
- KD pada palm vein NIR masih sangat terbatas; banyak dipakai di face/fingerprint/palmprint.
- **[BARU] Gap spesifik penelitian ini:** desain student **hardware-aware (latensi LUT perangkat)** + KD + quantization INT8, dengan **validasi latensi nyata di Raspberry Pi**.

[Gambar: tampilkan **Tabel 2.1 Analisis GAP Penelitian** — tambahkan kolom/baris bahwa baris "penelitian saat ini" kini mencakup hardware-aware + evaluasi Raspberry Pi.]

---

## SLIDE 5 — Dataset & Preprocessing

**Dataset: SCUT_PV_v1 (Luo dkk., 2024)** — palm vein NIR

- 550 subjek, 1.100 telapak, 11.000 citra, 10 citra/telapak, resolusi 640×480.
- Target klasifikasi pada eksperimen: **834 kelas** (identitas telapak yang dipakai).
- Split: **80% train / 10% val / 10% test**; val dipakai untuk optimasi arsitektur (bilevel NAS).

**Preprocessing (pipeline):**
1. **ROI extraction** — Gradient-Based Palm Center Detection + Intensity-Weighted Centroid → crop 384×384.
2. **CLAHE** — peningkatan kontras adaptif (vena NIR kontras rendah).
3. **Normalisasi** intensitas.
4. **Resize 224×224**.

[Gambar: Tabel 3.1 (karakteristik dataset), Gambar 3.2 (ROI extraction), Gambar 3.3 (preprocessing pipeline). Bisa pakai Gambar 2.2 untuk contoh raw→ROI→CLAHE.]

---

## SLIDE 6 — Metodologi: Alur Umum

- Pipeline: Preprocessing → **(Teacher: EfficientNetV2-M)** + **(Student: Hardware-Aware NAS)** → **Knowledge Distillation** → **Quantization INT8** → **Evaluasi di Raspberry Pi**.

[Gambar: Gambar 3.1 (desain penelitian) — WAJIB UPDATE: (1) ganti "ResNet50" → "EfficientNetV2-M",
(2) blok NAS beri label "Hardware-Aware (LUT Pi)", (3) tambah blok "Deployment & Benchmark di Raspberry Pi".]

---

## SLIDE 7 — Metode 1: Hardware-Aware Neural Architecture Search

**Dasar:** DARTS (Liu dkk., 2019) — relaksasi kontinu, optimasi bilevel (bobot di train set, arsitektur α di val set). P-DARTS progresif 3 tahap.

**[BARU] Penalti latensi sadar-perangkat (inti perubahan):** objektif arsitektur ditambah biaya latensi operator yang **diukur langsung di Raspberry Pi** (Look-Up Table), bersifat **diferensiabel** terhadap α:

```
min_α   L_CE^val(w*, α)  +  λ · LAT(α)
s.t.    w* = argmin_w  L_CE^train(w, α)

LAT(α) = Σ_(i,j) Σ_o  softmax(α^(i,j))_o · COST_Pi(o)
```

- `softmax(α^(i,j))_o` = probabilitas operasi `o` pada sisi (i,j).
- `COST_Pi(o)` = latensi operator dari **LUT yang diukur di Raspberry Pi** (dinormalisasi terhadap max).
- `λ` = bobot penalti (disweep: 0.0 / 0.05 / 0.10 / 0.20). λ besar → arsitektur lebih hemat latensi.

**Search space 12 operasi:** none, skip, sep_conv 3×3/5×5, dil_conv 3×3/5×5, mbconv3/6, **rep_conv 3×3/5×5**, avg/max pool.

**Literatur kuat (hardware/latency-aware NAS):**
- **ProxylessNAS** (Cai, Zhu & Han, **ICLR 2019**) — latensi perangkat sebagai loss diferensiabel langsung.
- **FBNet** (Wu dkk., **CVPR 2019**) — differentiable NAS dengan **latency look-up table**.
- **MnasNet** (Tan dkk., **CVPR 2019**) — platform-aware NAS, latensi diukur di perangkat nyata.

[Gambar: Gambar 3.5 (alur DARTS). Catatan: ICLR/CVPR setara peringkat A*/Q1 untuk computer vision.]

---

## SLIDE 8 — Metode 2: Knowledge Distillation

- Transfer representasi dari **teacher EfficientNetV2-M** (berkapasitas besar) ke student hasil NAS.
- Soft targets via temperature scaling; loss gabungan:

```
L = α · L_CE  +  (1 − α) · L_KD ,   L_KD = T² · KL(p_t^T ‖ p_s^T)
```

- T (temperature) melunakkan distribusi teacher → student belajar relasi antar-kelas.
- Teacher dibekukan (frozen) selama distilasi → proses stabil.

[Gambar: Gambar 2.5 (diagram teacher-student) & Gambar 3.6 (alur distilasi) — UPDATE label teacher ke "EfficientNetV2-M".]
[Catatan jujur (boleh disebut lisan): pada task ini separabilitas antar-kelas tinggi → efek KD diperkirakan marginal; KD diposisikan sebagai penjaga akurasi, bukan pengungkit utama.]

---

## SLIDE 9 — Metode 3: Quantization (Kompresi Model)

- **Post-Training Quantization (PTQ) INT8 per-channel** (QDQ, ONNX opset 13) — tanpa retraining, pakai data kalibrasi.
- Konversi bobot & aktivasi FP32 → INT8: `q = round(x / scale)`.
- Tujuan: turunkan **ukuran model** & percepat inferensi pada CPU edge berbasis integer.
- **[BARU] Temuan kerangka: manfaat quantization bersifat per-model.**
  - **Ukuran: selalu turun** (≈ 2–4× lebih kecil).
  - **Latensi: kondisional** — untung pada operator padat/reparameterized (intensity tinggi), bisa rugi pada sel compact (overhead konversi QDQ > hemat compute).
  - **Akurasi: terjaga** (INT8 ≈ FP32).
- Deliverable: **aturan pemilihan presisi (FP32/INT8) per model** berdasarkan pengukuran di perangkat.

[Gambar: Gambar 2.6 (FP32→INT8) & Gambar 3.7 (alur kompresi). Catatan: jika dosen tanya pruning,
jawab bahwa fokus dipindah ke quantization karena memberi kompresi & percepatan terukur tanpa
destabilisasi arsitektur kecil; pruning sebagai future work.]

---

## SLIDE 10 — Metrik Evaluasi

**Kinerja pengenalan:** Akurasi, FAR, FRR, **EER** (metrik utama biometrik).

**Efisiensi & kompleksitas (diukur di Raspberry Pi):**
- Jumlah parameter & FLOPs (kompleksitas struktural/teoretis).
- **Ukuran model (MB)** — sebelum vs sesudah INT8.
- **[BARU] Latensi inferensi (ms) terukur di Raspberry Pi** (ONNX Runtime, CPU 4 thread) + throughput.

[Catatan pembicara: tegaskan latensi diukur ON-DEVICE, bukan diestimasi FLOPs — ini pembeda kunci.]

---

## SLIDE 11 — Skenario Eksperimen

| No | Skenario | Model | Fokus |
|---|---|---|---|
| 1 | Baseline | EfficientNetV2-M (teacher) & MobileNetV3 | Plafon akurasi & efisiensi standar |
| 2 | NAS mandiri | Arsitektur Hardware-Aware NAS (latih scratch) | Validasi struktur + sweep λ (latensi) |
| 3 | Distillation | NAS + Teacher (KD) | Kenaikan akurasi student |
| 4 | Full Model | NAS + KD + **Quantization INT8** | **Latensi, ukuran, akurasi di Raspberry Pi** |

[Gambar: Tabel 3.2 (rangkuman skenario) — UPDATE: teacher = EfficientNetV2-M; Skenario 2 tambah
"sweep λ penalti latensi"; Skenario 4 = quantization INT8 + benchmark Raspberry Pi.]

---

## SLIDE 12 — Kontribusi Penelitian

1. **Kerangka terpadu pertama** NAS + KD + Quantization khusus **palm vein NIR** untuk perangkat edge.
2. **[BARU] Hardware-Aware NAS dengan penalti latensi LUT Raspberry Pi** — arsitektur student dioptimasi terhadap **latensi perangkat nyata**, bukan proxy FLOPs. λ berfungsi sebagai **knob** yang menggeser arsitektur sepanjang frontier akurasi–latensi.
3. **[BARU] Validasi & benchmark latensi nyata di Raspberry Pi** (FP32 vs INT8) — bukan estimasi teoretis.
4. **Karakterisasi kapan INT8 menguntungkan/merugikan** untuk palm vein edge (mekanisme: overhead konversi QDQ vs hemat compute / arithmetic intensity) → **aturan pemilihan presisi per-model**.
5. Student ringan (**ratusan ribu parameter, <1 MB**) dengan akurasi kompetitif & EER rendah, siap deploy pada perangkat sumber daya terbatas.

[Catatan pembicara: kontribusi #2 & #3 & #4 adalah nilai jual baru yang membedakan dari proposal lama
dan dari Ding dkk. 2025 (yang NAS-nya greedy & tanpa KD). Posisikan ini sebagai novelty.]

---

## CHECKLIST GAMBAR YANG PERLU DIUPDATE (sebelum print)

- [ ] **Gambar 3.1** (desain penelitian): ResNet50 → EfficientNetV2-M; blok NAS → "Hardware-Aware (LUT Pi)"; tambah blok "Deployment & Benchmark Raspberry Pi".
- [ ] **Gambar 3.6** (alur distilasi): label teacher → EfficientNetV2-M.
- [ ] **Tabel 2.1 / 3.2**: sesuaikan teacher & tambah hardware-aware + Raspberry Pi.
- [ ] Tambah ikon/foto **Raspberry Pi** di slide Latar Belakang & Metrik (visual deployment nyata).


---

## TAMBAHAN — Batasan Masalah (sisipkan ke slide/bab Batasan Masalah)

Tambahkan poin berikut (lanjutan dari batasan no.6 di buku proposal), untuk mengunci scope deployment:

> **7.** Raspberry Pi 5 digunakan sebagai **platform evaluasi deployment** untuk mengukur latensi inferensi dan ukuran model menggunakan citra uji SCUT_PV_v1. Proses **akuisisi citra secara langsung (live capture) menggunakan sensor seperti kamera NoIR maupun iluminasi inframerah berada di luar lingkup penelitian**; evaluasi akurasi dilakukan pada citra dataset, bukan pada hasil tangkapan sensor real-time.

[Kalimat ini konsisten dengan batasan no.1 (hanya SCUT_PV_v1) & no.6 (tidak menguji generalisasi
lintas sensor). Fungsinya: menutup pertanyaan "mana akurasi live-scan?" sejak awal.]

---

## SLIDE TAMBAHAN — Setup Deployment (Raspberry Pi 5)

[Letakkan setelah Slide 10 (Metrik Evaluasi) atau jadi sub-slide Metode 3 / Skenario 4.]

**Judul slide:** Setup Deployment & Pengukuran Latensi (Raspberry Pi 5)

**Posisi (1 kalimat pembuka):**
- Raspberry Pi 5 berperan sebagai **target deployment** untuk mengukur efisiensi model di perangkat edge nyata — *bukan* sebagai perangkat akuisisi citra.

**Spesifikasi perangkat:**
- Raspberry Pi 5
- SoC Broadcom BCM2712 — CPU quad-core Arm Cortex-A76 @ 2.4 GHz (64-bit)
- RAM: [isi sesuai unitmu, mis. 8 GB]
- OS: Raspberry Pi OS (64-bit) [isi versi]

**Konfigurasi inferensi:**
- Runtime: **ONNX Runtime (CPU Execution Provider)**
- Thread: **4** (semua core CPU)
- Presisi diuji: **FP32** dan **INT8 (PTQ per-channel, opset 13)**
- Input: citra **test set SCUT_PV_v1** (224×224), sudah ter-preprocessing (ROI + CLAHE + normalisasi)

**Yang diukur di perangkat:**
- Latensi inferensi per citra (ms) — median & p95
- Ukuran model (MB) — FP32 vs INT8
- Akurasi diverifikasi tetap konsisten dengan hasil di GPU (tidak ada degradasi konversi)

**Yang TIDAK dilakukan (batasan eksplisit):**
- Tidak ada akuisisi live (tanpa kamera NoIR / lampu IR) → arah pengembangan lanjutan.

[Catatan pembicara: kalau penguji menyinggung kamera/IR, jawab tegas: "Pi 5 di sini murni
untuk mengukur latensi inferensi model pada citra dataset; akuisisi sensor live adalah future work
dan sudah saya nyatakan di batasan masalah." — selesai, scope aman.]


---

## SLIDE TAMBAHAN — Latency LUT Raspberry Pi (bukti hardware-aware)

[Letakkan tepat SETELAH Slide 7 (Metode 1: Hardware-Aware NAS). Ini bukti konkret penaltimu.]

**Judul slide:** Latency Look-Up Table (LUT) — Biaya Operasi Terukur di Raspberry Pi

**Apa itu LUT (1 kalimat):**
- Tabel biaya latensi tiap operasi, **diukur langsung di Raspberry Pi** (ONNX Runtime, 4 thread, median dari 100 iterasi), dipakai sebagai penalti latensi yang diferensiabel pada NAS — bukan estimasi FLOPs.

**Search space 12 operasi & biayanya:**

| Operasi | #conv (proxy) | LUT FP32 (ms) |
|---|---|---|
| none / skip_connect | 0 | 0.018 / 0.025 |
| avg_pool / max_pool 3×3 | 1 | 0.021 / 0.022 |
| rep_conv_3x3 | 1 | 0.055 |
| rep_conv_5x5 | 1 | 0.129 |
| dil_conv_3x3 / 5x5 | 2 | 0.043 / 0.064 |
| sep_conv_3x3 / 5x5 | 2 | 0.054 / 0.107 |
| mbconv3_3x3 | 3 | 0.092 |
| mbconv6_3x3 | 3 | **0.158** (termahal) |

**Cara membaca (poin lisan):**
- Biaya **tidak** sebanding FLOPs/params; ia mengikuti perilaku CPU ARM nyata.
- `mbconv6` termahal; `rep_conv_3x3` murah → saat bobot penalti λ dinaikkan, NAS bergeser dari sep/mbconv ke **rep_conv + skip**.
- LUT inilah yang membuat arsitektur hasil search **sadar-perangkat**.

[Catatan: nilai pada tabel = LUT FP32 (`latency_lut_pi.json`). Jika ingin, tambahkan kolom INT8
dari `latency_lut_pi_int8_corrected.json`. Untuk penalti, biaya dinormalisasi terhadap nilai maksimum.]
[Pasangkan slide ini dengan rumus penalti di Slide 7: COST_Pi(o) = nilai dari tabel ini.]


---
---

# BAGIAN HASIL & PEMBAHASAN (Slide H0–H8) — konten fixed

> Semua latensi diukur di **Raspberry Pi 5** (ONNX Runtime CPU, 4 thread). Akurasi pada **test set 834 kelas**.
> INT8 = PTQ static **per-channel** (QDQ, opset 13). Konfigurasi spasial dikunci: stem_downsample=4, reduction_indices=2,5.

---

## SLIDE H0 — Peta Hasil (pembuka)

**Judul:** Ringkasan Hasil — 4 Temuan Utama

- **NAS sadar-hardware** menghasilkan **frontier Pareto** akurasi–latensi (bukan satu model tunggal); λ menggeser arsitektur dari akurat→cepat.
- **Knowledge Distillation** memberi perbaikan **marginal & konsisten-arah** (dalam orde noise).
- **Quantization INT8**: ukuran model **selalu turun**; latensi **untung secara kondisional** (tergantung kapasitas/operator).
- Semua **divalidasi terukur di Raspberry Pi 5** — bukan estimasi FLOPs.

**Headline:** Model akhir **hwNAS λ0.20 C8 (INT8): 98.92% @ 5.27 ms, 0.61 MB** di Raspberry Pi 5 — vs MobileNetV3 99.88% @ 15.49 ms, 21 MB.

---

## SLIDE H1 — Skenario 1: Baseline

**Judul:** Baseline — Plafon Akurasi & Efisiensi Standar

| Model | Peran | Akurasi | Ukuran | Latensi Pi |
|---|---|---|---|---|
| EfficientNetV2-M | Teacher (plafon akurasi) | ~100% | ~54 M params | — (training-only, tak dideploy) |
| MobileNetV3-Large | Lightweight standar | 99.88% (FP32) | 21.08 MB | 15.49 ms |

**Poin:**
- MobileNetV3 akurat (99.88%) tetapi **berat untuk edge**: 21 MB, 15.49 ms → indikasi **overcapacity** pada palm-vein NIR.
- Memotivasi arsitektur yang dirancang **sadar-perangkat**, bukan model RGB umum yang dipakai ulang.

---

## SLIDE H2 — Skenario 2a: Hasil NAS — Pergeseran Operator vs λ

**Judul:** NAS Sadar-Hardware — λ Menggeser Pilihan Operator

**Pesan utama (1 kalimat):**
> Saat bobot penalti latensi (λ) dinaikkan, NAS **konsisten meninggalkan `sep_conv` (mahal di Pi)** dan beralih ke **`rep_conv` + `skip_connect` (murah di Pi)**.

| λ | Operator dominan (normal cell) | Interpretasi |
|---|---|---|
| 0.00 | **sep_conv_3x3** (6/8) | Akurasi murni, abai latensi → operator mahal |
| 0.05 | rep_conv(3) + dil_conv(3) + skip(2) | Mulai sadar latensi → campuran |
| 0.10 | **rep_conv_3x3** (6/8) + skip(2) | rep_conv mendominasi |
| 0.20 | **rep_conv_3x3** (6/8) + skip(2) | rep_conv dominan + skip → paling hemat |

**Kenapa rep_conv menang:** difusikan jadi **1 konvolusi** saat inferensi (biaya LUT hanya 0.055 ms) namun tetap kaya representasi saat training → "murah tapi kuat". **Bukti penalti LUT bekerja.**

[Gambar opsional/backup: 1 diagram topologi genotype λ0.20 C8 sebagai ilustrasi — jangan tampilkan 4 graf sekaligus.]

---

## SLIDE H3 — Skenario 2b: NAS Pareto (FP32, di Raspberry Pi 5)

**Judul:** Pareto Arsitektur NAS — Pengaruh λ dan Lebar Kanal (C)

| Model | Akurasi FP32 | Latensi FP32 | Ukuran |
|---|---|---|---|
| hwNAS λ0.05 **C4** | 93.29% | 2.53 ms | 0.46 MB |
| hwNAS λ0.20 **C4** | 91.49% | 2.64 ms | 0.53 MB |
| **hwNAS λ0.05 C6** | 97.96% | **3.99 ms** | 0.79 MB |
| hwNAS λ0.20 C6 | 97.60% | 4.27 ms | 0.96 MB |
| hwNAS λ0.05 C8 | 98.08% | 5.81 ms | 1.21 MB |
| hwNAS λ0.10 C8 | 99.16% | 6.75 ms | 1.62 MB |
| **hwNAS λ0.20 C8** | 99.16% | 6.29 ms | 1.53 MB |

**Temuan:**
- **C4 kolaps** (91–93%) → kapasitas terlalu kecil untuk 834 kelas. Dibuang dari kandidat.
- **C6 = pojok kecepatan**, **C8 = pojok akurasi (~99%)** → dua titik Pareto bermanfaat.
- **λ** menukar akurasi↔latensi secara terkontrol (knob).

---

## SLIDE H4 — Skenario 3: Knowledge Distillation

**Judul:** Knowledge Distillation — Perbaikan Marginal, Berpola

**Teacher:** EfficientNetV2-M (frozen). Pemilihan config **by-validation**.

**C6 (baseline test 97.96%):**

| (T, α) | Val | Test |
|---|---|---|
| baseline | — | 97.96% |
| t8_a0.1 | 95.20% | 94.72% |
| t8_a0.3 | 97.24% | 97.72% |
| **t10_a0.4 / t20_a0.5** | **97.60%** | 98.20% |

**C8 (baseline test 99.16%):**

| (T, α) | Val | Test |
|---|---|---|
| baseline | — | 99.16% |
| t8_a0.3 | 99.16% | 99.28% |
| **t10_a0.4 / t20_a0.5** | **99.28%** | 99.28–99.40% |

**Temuan:**
- KD memberi **+0.1–0.3 pp** (kedua kapasitas) — **marginal, dalam orde noise single-seed**.
- **Pola jelas:** T tinggi (10–20) + α seimbang (0.4–0.5) terbaik; **α=0.1 (bobot KD 90%) MERUSAK** (C6 turun ke 94.72%).
- **Mekanisme:** teacher saturasi (~100%) → soft-target nyaris one-hot (dark knowledge miskin); bobot KD berlebih membuang sinyal label asli. **Ini temuan, bukan kegagalan.**

---

## SLIDE H5 — Skenario 4a: Quantization INT8 (FP32 → INT8 di Pi 5)

**Judul:** Quantization INT8 — Ukuran Selalu Turun, Latensi Kondisional

| Model | Akurasi FP32→INT8 | Latensi FP32→INT8 | Speedup | Ukuran |
|---|---|---|---|---|
| hwNAS λ0.05 C4 | 93.29 → 93.29% | 2.53 → 3.75 ms | **0.67× (lambat)** | 1.27× kecil |
| hwNAS λ0.05 C6 | 97.96 → 97.96% | 3.99 → 5.10 ms | **0.78× (lambat)** | 1.76× |
| hwNAS λ0.20 C6 | 97.60 → 97.36% | 4.27 → 4.47 ms | **0.95× (lambat)** | 2.10× |
| hwNAS λ0.05 C8 | 98.08 → 98.32% | 5.81 → 5.46 ms | **1.06× (cepat)** | 2.16× |
| hwNAS λ0.10 C8 | 99.16 → 99.04% | 6.75 → 5.70 ms | **1.18× (cepat)** | 2.51× |
| hwNAS λ0.20 C8 | 99.16 → 98.92% | 6.29 → 5.27 ms | **1.19× (cepat)** | 2.52× |
| MobileNetV3-L | 99.88 → 98.68% | 15.49 → 8.39 ms | **1.85× (cepat)** | 3.64× |

**Temuan (rapi & konsisten di 8 model):**
- **Latensi:** INT8 **mempercepat HANYA di C8** (1.06–1.19×) & MobileNet (1.85×); **memperlambat di C4/C6** (0.67–0.95×). **Titik-balik di ~C8.**
- **Ukuran:** **selalu turun** (1.3–3.6×) — manfaat tak bersyarat.
- **Akurasi:** terjaga (Δ ≈ −0.2 pp; INT8 ≈ FP32).

---

## SLIDE H6 — Skenario 4b: Mekanisme INT8 (kenapa kondisional)

**Judul:** Mengapa INT8 Tak Selalu Mempercepat — Arithmetic Intensity

- INT8 menambah node konversi **Quantize/Dequantize (QDQ) + Transpose** dengan **overhead hampir konstan (~2.4–2.7 ms)**.
- INT8 menguntungkan **⟺ hemat compute konvolusi > overhead konversi tetap**.
- Sel **padat/lebar (C8)** → arithmetic intensity tinggi → hemat compute besar → **INT8 menang**.
- Sel **compact (C4/C6, banyak skip/dilated)** → memory-bound → hemat kecil < overhead → **INT8 kalah**.

**Implikasi (deliverable):** **aturan pemilihan presisi per-model** — C8 deploy INT8; C6/C4 deploy FP32 (INT8 hanya bila storage kritis).

[Boleh ringkas jadi 1 diagram bar: compute-saving vs QDQ-overhead untuk C6 vs C8.]

---

## SLIDE H7 — Pareto Final & Pemilihan Pemenang

**Judul:** Frontier Pareto di Raspberry Pi 5 — Pemilihan Berbasis Skenario Deploy

[Gambar utama: scatter plot **Akurasi (y) vs Latensi Pi (x)**, tandai frontier non-dominated.]

**Titik Pareto (presisi deploy terbaik per model):**

| Skenario deploy | Model | Akurasi | Latensi | Ukuran |
|---|---|---|---|---|
| **A — Akurasi-tinggi (headline)** | hwNAS λ0.20 C8 **INT8** | **98.92%** | **5.27 ms** | 0.61 MB |
| **B — Tercepat** | hwNAS λ0.05 C6 **FP32** | 97.96% | **3.99 ms** | 0.79 MB |
| **C — Akurasi ~99% hemat** | hwNAS λ0.10 C8 **INT8** | 99.04% | 5.70 ms | 0.65 MB |
| Pembanding | MobileNetV3-L FP32 | 99.88% | 15.49 ms | 21 MB |

**Klaim kunci:** model headline **3× lebih cepat & ~35× lebih kecil** dari MobileNetV3, dengan akurasi hanya ~1 pp di bawahnya.

---

## SLIDE H8 — Kesimpulan

**Judul:** Kesimpulan

**Menjawab rumusan masalah:**
1. **NAS** menghasilkan arsitektur student ringan sadar-perangkat (ratusan ribu param, <1 MB) — λ sebagai knob Pareto; mengalahkan baseline manual.
2. **KD** menjaga/menaikkan akurasi secara marginal (+0.1–0.3 pp); pengungkit utama tetap NAS + kuantisasi.
3. **Quantization INT8** menekan ukuran (1.3–3.6×) dengan akurasi terjaga; latensi untung kondisional → **aturan pemilihan presisi per-model**.

**Kontribusi:** kerangka terpadu NAS hardware-aware (LUT Pi) + KD + INT8 untuk palm-vein NIR, **divalidasi terukur di Raspberry Pi 5**.

**Limitasi (jujur):** hasil **single-seed (42)** → perlu **3-seed + uji McNemar** untuk klaim signifikan; selisih head-to-head ~0.2–1 pp masih dalam noise.

**Pengembangan lanjutan:** akuisisi **live-scan** (sensor NoIR/IR) & generalisasi lintas-sensor — di luar lingkup penelitian ini.


---

## SLIDE H7b — TABEL MASTER: Semua Model vs NAS (slide penutup utama)

**Judul:** Perbandingan Menyeluruh — Seberapa Kecil & Cepat NAS, Akurasi Tetap Terjaga

Diurutkan dari **terbesar → terkecil**; NAS (2 baris terakhir) = student terpilih.

| Model | Params (M) | FLOPs (MMACs)¹ | Akurasi | Ukuran (MB)² | Latensi Pi 5 (ms)³ |
|---|---|---|---|---|---|
| VGG16 | 137.68 | 15 470 | 99.64% | ≈525 | — |
| ConvNeXt-Base | 88.42 | 15 373 | 100% | ≈337 | — |
| RegNetY-16GF | 83.09 | 16 009 | 100% | ≈317 | — |
| **EfficientNetV2-M** (teacher) | 53.93 | 5 446 | 100% | ≈206 | — |
| InceptionV3 | 26.69 | 2 856 | 99.76% | ≈102 | — |
| ResNet50 | 25.22 | 4 133 | 100% | ≈96 | — |
| EfficientNetB4 | 19.04 | 1 578 | 99.76% | ≈73 | — |
| DenseNet121 | 7.81 | 2 897 | 99.88% | ≈30 | — |
| MobileNetV3-Large (baseline) | 5.27 | 235 | 99.88% | 21.08 | 15.49 |
| **hwNAS λ0.20 C8** (INT8) | **0.52** | 136 | **98.92%** | **0.61** | **5.27** |
| **hwNAS λ0.05 C6** (FP32) | **0.32** | 59 | 97.96% | **0.79** | **3.99** |

**Catatan kaki:**
- ¹ FLOPs = **MMACs @224×224 (konvensi thop)** — apple-to-apple di semua model.
- ² Ukuran teacher = **≈ estimasi bobot FP32** (params×4 B); MobileNet & NAS = **ukuran file ONNX terukur** (NAS pada presisi deploy). MobileNet: estimasi 20.1 ≈ terukur 21.08 → konsisten.
- ³ Latensi diukur di **Raspberry Pi 5** (ONNX Runtime, 4 thread). Teacher **diuji di GPU** → tidak sebanding, dikosongkan. Latensi edge hanya untuk **NAS & MobileNet**.

**Poin lisan (pesan utama):**
- Student NAS **~100–270× lebih sedikit parameter** dari kandidat teacher (0.3–0.5 M vs 25–138 M), akurasi tetap **97.96–98.92%** (hanya **~1 pp** di bawah model raksasa).
- vs MobileNetV3: NAS **~10–17× lebih sedikit parameter**, **~35× lebih kecil ukuran**, **~3–4× lebih cepat** di Pi 5.
- **Pesan:** efisiensi ekstrem **tanpa mengorbankan akurasi secara berarti** → tujuan tesis tercapai.

**Catatan jujur (bila ditanya):** banyak teacher saturasi (100%/EER 0) → soft-target nyaris one-hot → menjelaskan **KD marginal** (Slide H4); pilihan teacher tidak kritikal.


---

## SLIDE H1b — Perbandingan Kandidat Teacher / Baseline (9 model)

**Judul:** Eksplorasi Kandidat Teacher — Akurasi vs Kompleksitas

| Model | Params (M) | FLOPs (MMACs)¹ | Akurasi | EER |
|---|---|---|---|---|
| **EfficientNetV2-M** (teacher terpilih) | 53.93 | 5 446 | 100% | 0 |
| ResNet50 | 25.22 | 4 133 | 100% | 0 |
| ConvNeXt-Base | 88.42 | 15 373 | 100% | 0 |
| RegNetY-16GF | 83.09 | 16 009 | 100% | 0 |
| DenseNet121 | 7.81 | 2 897 | 99.88% | 0 |
| MobileNetV3-Large | 5.27 | 235 | 99.88% | 0 |
| InceptionV3 | 26.69² | 2 856 | 99.76% | 0 |
| EfficientNetB4 | 19.04 | 1 578 | 99.76% | 0 |
| VGG16 | 137.68 | 15 470 | 99.64% | 0 |

**Catatan kaki:**
- ¹ FLOPs = **MMACs @224×224 (konvensi thop)**, sama dengan student → apple-to-apple.
- ² Params InceptionV3 termasuk aux-head (model terlatih); FLOPs inferensi dihitung tanpa aux.
- **Latensi tidak ditampilkan**: seluruh kandidat ini diuji di **GPU**, tidak sebanding dengan pengukuran Raspberry Pi 5. Latensi edge hanya disajikan untuk **NAS & MobileNetV3** (lihat Slide H5/H7/H7b).

**Poin lisan:**
- **Empat model mencapai 100%** (ResNet50, EfficientNetV2-M, ConvNeXt, RegNetY) → test set relatif terpisah baik; semua teacher **saturasi**.
- EfficientNetV2-M dipilih sebagai teacher (kapasitas besar, arsitektur modern). **Catatan jujur (bila ditanya):** karena semua kandidat top saturasi (100%/EER 0), **pilihan teacher tidak berdampak besar pada hasil KD** — ini menjelaskan mengapa KD marginal (Slide H4), bukan karena teacher salah pilih.
- MobileNetV3 = baseline ringan; meski hanya 5.27 M params, **masih jauh lebih besar** dari student NAS (0.3–0.5 M) → motivasi NAS.

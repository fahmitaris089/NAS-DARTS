# Update PPT Pra-Sidang — Hasil Eksperimen & Revisi Slide

> Sumber angka: file `test_results.json` / `eval_results.json` / `benchmark_int8_static_results.json`
> / `search_summary.json` di repo. Pi latency = **pending** (belum diukur live).

---

## 1. ANGKA HASIL (ground truth)

### 1.1 NAS Search (3 run)
| Run | Operasi dominan | Est. Params | Waktu Search |
|---|---|---|---|
| `search` (DARTS asli) | dil_conv 3x3/5x5 | ~316K | 237 min |
| `search_mobile_v1` | sep_conv 3x3/5x5 | ~351K | 298 min |
| `search_mobile_v2` (TOPOLOGI DIPAKAI) | mbconv3 + skip | — | 129 min |

### 1.2 Retrain Student (834 kelas, test 834 sampel)
| Model | Operasi | Acc | F1 | EER | Params | FLOPs | Size FP32→INT8 | Lat. CPU desktop | Lat. Pi 5 |
|---|---|---|---|---|---|---|---|---|---|
| NAS v1 C4 | sep_conv | 93.65% | 0.917 | 0.061% | 0.21M | 97.8M | 0.46→0.56 MB | 16.4 ms | pending |
| NAS v2 C4 | mbconv | 98.08% | 0.975 | 0.003% | 0.24M | 184.8M | 0.56→0.59 MB | 16.9 ms | pending |
| **NAS rep_conv C8 (FINAL)** | rep_conv | **98.80%** | **0.984** | **0.002%** | 0.50M | 130.1M | **1.46→0.60 MB** | 12.8 ms | pending |

### 1.3 Benchmark Teacher / Baseline (9 model, `Teacher/training_results`)
| Model | Acc | F1 | Params | Inferensi (ms) | Train (min) | Catatan |
|---|---|---|---|---|---|---|
| ResNet50 | 100% | 1.000 | 25.2M | 146.2 | 72.3 | **Teacher batas-atas** (acc tertinggi, train tercepat di kelas top) |
| EfficientNetV2-M | 100% | 1.000 | 53.9M | 149.2 | 97.9 | **Teacher KD aktual** |
| ConvNeXtBase | 100% | 1.000 | 88.4M | 163.6 | 150.9 | |
| RegNetY16GF | 100% | 1.000 | 83.1M | 168.8 | 125.3 | |
| DenseNet121 | 99.88% | 0.998 | 7.8M | 147.6 | 80.6 | |
| MobileNetV3-Large | 99.88% | 0.998 | 5.3M | 145.6 | 64.6 | **Baseline ringan manual** |
| InceptionV3 | 99.76% | 0.997 | 26.7M | 153.4 | 88.5 | |
| EfficientNetB4 | 99.76% | 0.997 | 19.0M | 150.7 | 85.4 | |
| VGG16 | 99.64% | 0.995 | 137.7M | 154.6 | 94.1 | terbesar, akurasi terendah |

Insight: 4 model tembus 100% (ResNet50, EffNetV2-M, ConvNeXt, RegNet). ResNet50 dipilih sebagai
teacher batas-atas karena akurasi 100% dengan params & waktu latih paling efisien di tier teratas.

### 1.4 Knowledge Distillation (teacher = EfficientNetV2-M, student = mbconv v2)
| Student | Tanpa KD (NOKD) | Dengan KD | Gain | Catatan |
|---|---|---|---|---|
| v2 C3 | 96.04% | 97.00% | **+0.96%** | student kecil → KD lebih berdampak |
| v2 C4 | 98.56% | 98.92% | +0.36% | EER naik 0.0009→0.0066%; NOKD 150 ep vs KD 500 ep |

### 1.5 Pi per-operasi (LUT, 4 thread) — justifikasi pilih rep_conv
`rep_conv_3x3 = 0.055 ms` ≈ `sep_conv_3x3 = 0.054 ms`, jauh < `mbconv3 = 0.092 ms`, `mbconv6 = 0.158 ms`.

---

## 2. REVISI SLIDE LAMA (14 slide)

| Slide | Aksi | Detail |
|---|---|---|
| 1 Judul | opsional | tambah "+ Raspberry Pi 5 Deployment" |
| 2 Latar Belakang | keep | — |
| 3 Prototipe | keep | — |
| 4 Fokus | keep | — |
| 5 Masalah & Solusi | **EDIT** | kartu Latensi: hapus "Pruning", ganti "INT8 PTQ + rep_conv reparam + Deploy Pi 5" |
| 6 Gap/Novelty | **EDIT** | poin 3 novelty: ganti "Hybrid Compression (Pruning+Quant)" → "rep_conv NAS-designed + INT8 PTQ untuk edge NIR" |
| 7 Dataset | cek | luruskan jumlah kelas (deck 550/1100 vs eksperimen 834 kelas) |
| 8 NAS/DARTS | **EDIT** | tambah `rep_conv 3x3/5x5` ke search space |
| 9 KD | **EDIT PENTING** | teacher = **EfficientNetV2-M** (BUKAN ResNet50). Perbaiki diagram & teks |
| 10 Kompresi | **EDIT besar** | hapus Pruning. Isi: (1) rep_conv reparam, (2) INT8 PTQ 1.46→0.60 MB (~2.4x) |
| 11 Metrik | keep | — |
| 12 Skenario | **EDIT** | hapus Pruning dari skenario 4 → "Student + INT8 + Deploy Pi 5"; tambah skenario operasi (sep/mbconv/rep_conv) |
| 13 Kontribusi | **EDIT** | badge: "INT8 Quantized" + "Raspberry Pi Ready" (hapus Pruning) |
| 14 Penutup | keep | — |

---

## 3. SLIDE BARU (sisip setelah slide 12) — INTI HASIL

### Slide A — Hasil NAS: Pemilihan Operasi
Judul: **"Pengaruh Operasi pada Arsitektur Student"**
- Tabel 1.2 (baris v1/v2/rep_conv): Acc, Params, FLOPs.
- Insight: "Topologi v2 (mbconv) akurasi tinggi; operasi diganti **rep_conv** agar murah di edge (FLOPs 185M→130M) sambil mempertahankan akurasi (98.08%→98.80%)."

### Slide B — Komparasi Model Lengkap
Judul: **"Perbandingan Menyeluruh: Akurasi vs Efisiensi"** → pakai tabel 1.2 + 1.3 digabung.
Kolom: Peran | Acc | F1 | EER | Params | FLOPs | Size FP32→INT8 | Lat. Pi 5 (pending).

### Slide B2 — Benchmark Teacher (9 Model Besar)
Judul: **"Penentuan Batas-Atas: Benchmark 9 Arsitektur Teacher"** → pakai tabel 1.3.
- Pesan: "ResNet50 dipilih sebagai teacher referensi (100% acc, params & waktu latih paling efisien di tier 100%); EfficientNetV2-M dipakai sebagai teacher distilasi."
- Bisa pakai grafik `Teacher/training_results/accuracy_comparison.png` jika mau visual.

### Slide C — Knowledge Distillation
Judul: **"Knowledge Distillation: Knowledge Gain"** → pakai tabel 1.4.
- Teacher: EfficientNetV2-M. Pesan: "KD memberi gain lebih besar pada student berkapasitas kecil (C3 +0.96%)."

### Slide D — Efek INT8 Quantization
"rep_conv: 1.46 MB (FP32) → **0.60 MB** (INT8), ~2.4x lebih kecil; ~27x lebih kecil dari MobileNetV3-L (16 MB) dengan selisih akurasi ~1%."

### Slide E — Takeaway / Kontribusi terukur
"NAS rep_conv = 98.80% (gap 1.2% dari Teacher 100%), params **50x lebih kecil** dari ResNet50 & **10x** dari MobileNetV3-L, EER 0.002%, deploy-ready INT8 0.60 MB di Raspberry Pi 5 (latency pengukuran live)."

---

## 4. FLAGS PENTING (antisipasi pertanyaan penguji)

1. **Latency Pi belum terukur** — tulis "pengukuran live sedang berjalan". Jangan kasih angka.
2. **Teacher KD = EfficientNetV2-M, bukan ResNet50** — konsistenkan semua slide.
3. **KD ≠ pada model final** — KD diuji pada student mbconv v2, bukan rep_conv. Jangan klaim "rep_conv + KD + INT8" sebagai satu model end-to-end.
4. **Confound epoch di KD C4** (NOKD 150 ep vs KD 500 ep) — gain +0.36% belum bersih; pakai narasi C3 (+0.96%) sebagai bukti utama KD bekerja.
5. **Test set tak setara**: student diuji 834 sampel; baseline_mobilenetv3 folder nas_results hanya 20 sampel/2 kelas → JANGAN dipakai. Gunakan MobileNetV3-Large dari folder Teacher.
6. **Pruning sudah dihapus** dari metodologi — pastikan tidak ada sisa klaim Pruning di slide manapun.

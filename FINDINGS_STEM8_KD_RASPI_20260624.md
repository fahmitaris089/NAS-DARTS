# Findings Eksperimen Terbaru: Stem-8 Retrain, KD, dan Benchmark Raspberry Pi

Dokumen ini merangkum temuan terbaru setelah update eksperimen `stem_downsample=8` pada hasil search `search_hwint8_l0.05` dan `search_hwint8_l0.20`, plus evaluasi Raspberry Pi dari model retrain dan KD.

Semua akurasi mengacu ke test set 834 kelas. Semua hasil Raspberry Pi mengacu ke ONNX Runtime CPU, 4 threads, dan evaluasi FP32 vs INT8 static PTQ.

## Ringkasan Inti

1. Untuk seri retrain `stemds8`, kapasitas model tetap menjadi pengendali utama akurasi: `C10 > C8 > C6`.
2. Pada keluarga `lambda=0.20`, baseline retrain yang paling masuk akal untuk dibawa ke tahap KD adalah `retrain_hwNAS_L0.20_C8_stemds8_834cls`, karena berada di titik tengah yang baik antara akurasi, ukuran, dan latency Pi.
3. KD memang memberi peningkatan nyata terhadap baseline retrain `L0.20 C8 stemds8`, tetapi peningkatannya kecil-moderat, bukan lonjakan besar.
4. Konfigurasi KD terbaik dari eksperimen terbaru adalah `kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed`, karena mencapai akurasi tertinggi sambil tetap mempertahankan profil deploy Pi yang hampir sama dengan model retrain asalnya.
5. Di Raspberry Pi, seluruh model NAS/KD yang diuji tetap kecil dan cepat. INT8 konsisten mengecilkan ukuran model dan umumnya mempercepat inferensi untuk konfigurasi C8/C10, dengan penalty akurasi yang kecil.

## A. Retrain Stem-8: Pola Utama

Semua model retrain memakai `num_cells=8`, `stem_downsample=8`, `reduction_indices=2,5`. Perbedaan utama ada pada `lambda` hasil search dan `C_init`.

### Retrain `lambda=0.05`

| Model | Test Acc | Params | FLOPs | Pi FP32 | Pi INT8 |
|---|---:|---:|---:|---:|---:|
| `retrain_hwNAS_L0.05_C6_stemds8_834cls` | 96.52% | 315,754 | 19.97M | 2.05 ms | 2.36 ms |
| `retrain_hwNAS_L0.05_C8_stemds8_834cls` | 98.08% | 432,764 | 31.58M | 2.87 ms | 2.43 ms |
| `retrain_hwNAS_L0.05_C10_stemds8_834cls` | 99.04% | 573,766 | 45.67M | 3.52 ms | 3.17 ms |

Temuan:
- `C6` terlalu kecil untuk jadi kandidat utama karena akurasinya tertinggal cukup jauh.
- `C8` memberi trade-off yang lebih seimbang.
- `C10` adalah pemenang akurasi pada cabang `lambda=0.05`, dan masih sangat ringan untuk ukuran edge model.

### Retrain `lambda=0.20`

| Model | Test Acc | Params | FLOPs | Pi FP32 | Pi INT8 |
|---|---:|---:|---:|---:|---:|
| `retrain_hwNAS_L0.20_C6_stemds8_834cls` | 97.72% | 365,722 | 25.29M | 2.17 ms | 2.09 ms |
| `retrain_hwNAS_L0.20_C8_stemds8_834cls` | 98.44% | 522,332 | 41.09M | 3.00 ms | 2.22 ms |
| `retrain_hwNAS_L0.20_C10_stemds8_834cls` | 98.80% | 714,406 | 60.56M | 3.77 ms | 2.94 ms |

Temuan:
- `lambda=0.20` memberi model yang tetap efisien, tetapi di set ini akurasinya tidak melampaui `lambda=0.05 C10`.
- `L0.20 C8` terlihat sebagai titik tengah yang rapi: akurasi lebih tinggi dari `C6`, tetapi latency/size masih sangat rendah dibanding C10 dan jauh di bawah baseline CNN besar.
- Karena itu, `retrain_hwNAS_L0.20_C8_stemds8_834cls` layak dijadikan baseline utama untuk eksperimen KD.

## B. KD di Atas Baseline `L0.20 C8 stemds8`

Baseline retrain yang dibandingkan:

- `retrain_hwNAS_L0.20_C8_stemds8_834cls`
- Test accuracy: **98.44%**
- Pi FP32: **3.00 ms**
- Pi INT8: **2.22 ms**
- INT8 size: **0.602 MB**

### Hasil KD / fine-tune

| Model | T | alpha | KD weight | Test Acc | Pi FP32 | Pi INT8 |
|---|---:|---:|---:|---:|---:|---:|
| `finetune_hwNAS_L0.20_C8_t3_a1.0_ls0_nomix_lr1e4_fixed` | 3 | 1.0 | 0.0 | 98.20% | 2.97 ms | 2.28 ms |
| `kd_hwNAS_L0.20_C8_t2_a0.8_ls0_nomix_lr1e4_fixed` | 2 | 0.8 | 0.2 | 98.92% | 2.98 ms | 2.30 ms |
| `kd_hwNAS_L0.20_C8_t3_a0.7_ls0_nomix_lr1e4_fixed` | 3 | 0.7 | 0.3 | 99.04% | 2.95 ms | 2.29 ms |
| `kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed` | 3 | 0.8 | 0.2 | 99.04% | 3.00 ms | 2.27 ms |
| `kd_hwNAS_L0.20_C8_t3_a0.9_ls0_nomix_lr1e4_fixed` | 3 | 0.9 | 0.1 | 98.68% | 2.95 ms | 2.26 ms |
| `kd_hwNAS_L0.20_C8_t4_a0.8_ls0_nomix_lr1e4_fixed` | 4 | 0.8 | 0.2 | 98.92% | 2.99 ms | 2.33 ms |

Temuan:
- KD memang membantu dibanding baseline retrain `98.44%`.
- Konfigurasi tanpa KD murni (`alpha=1.0`, `kd_weight=0.0`) justru turun ke `98.20%`. Ini penting: peningkatan bukan berasal dari fine-tuning biasa, tetapi dari sinyal distillation.
- Konfigurasi terbaik adalah dua kandidat:
  - `t3, alpha=0.7`
  - `t3, alpha=0.8`
- Keduanya mencapai `99.04%`, tetapi `t3, alpha=0.8` lebih aman sebagai kandidat utama karena:
  - performanya sama dengan yang terbaik,
  - bobot hard-label vs soft-target lebih seimbang,
  - sebelumnya juga sudah menjadi kandidat baseline yang Anda anggap paling representatif.

## C. Implikasi KD terhadap Baseline Utama

Perbandingan paling penting sekarang adalah:

| Model | Test Acc | Pi FP32 | Pi INT8 | INT8 Size |
|---|---:|---:|---:|---:|
| `retrain_hwNAS_L0.20_C8_stemds8_834cls` | 98.44% | 3.00 ms | 2.22 ms | 0.602 MB |
| `kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed` | 99.04% | 3.00 ms | 2.27 ms | 0.596 MB |

Interpretasi:
- KD memberi **+0.60 pp** akurasi test terhadap baseline retrain.
- Latency Pi praktis tidak berubah:
  - FP32 tetap `3.00 ms`
  - INT8 hanya bergeser `2.22 ms -> 2.27 ms`
- Ukuran INT8 juga praktis identik, bahkan sedikit lebih kecil pada model KD (`0.596 MB` vs `0.602 MB`).

Kesimpulan praktis:
- Jika tujuan utama adalah memilih **satu model headline hasil NAS+KD+INT8**, maka `kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed` sekarang adalah kandidat paling kuat.
- Jika tujuan utama adalah menjaga narasi metodologi tetap bersih, maka pasangan yang paling tepat untuk dibandingkan adalah:
  - baseline retrain: `retrain_hwNAS_L0.20_C8_stemds8_834cls`
  - model final KD: `kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed`

## D. Posisi Hasil Retrain vs KD

Ada dua cara framing yang valid, tergantung tujuan tulisan:

### Opsi 1: Fokus pada baseline method pipeline

Gunakan `retrain_hwNAS_L0.20_C8_stemds8_834cls` sebagai baseline internal NAS, lalu tunjukkan bahwa KD meningkatkan akurasi tanpa merusak efisiensi deploy secara berarti.

Narasi:
> Search hardware-aware menghasilkan baseline retrain C8 yang sudah efisien di Raspberry Pi. Distillation dari EfficientNetV2-M kemudian meningkatkan akurasi dari 98.44% menjadi 99.04%, sementara latency dan ukuran model tetap hampir sama.

### Opsi 2: Fokus pada model terbaik murni

Jika yang dicari adalah model terbaik seluruh eksperimen terbaru, maka:
- retrain terbaik = `retrain_hwNAS_L0.05_C10_stemds8_834cls` pada 99.04%
- KD terbaik = `kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed` atau `t3_a0.7` pada 99.04%

Tetapi untuk narasi penelitian, opsi ini lebih sulit karena membandingkan dua search branch berbeda (`lambda=0.05` vs `lambda=0.20`) dan tidak menunjukkan langsung efek KD pada satu baseline yang sama.

Karena itu, untuk pembahasan metodologi, opsi 1 lebih kuat.

## E. Temuan Raspberry Pi

### 1. Seluruh model NAS/KD tetap edge-feasible

Semua model stem-8 dan KD yang diuji berada di kisaran:
- FP32 mean latency: `2.05 - 3.77 ms`
- INT8 mean latency: `2.09 - 3.17 ms`
- INT8 size: `0.450 - 0.782 MB`

Ini jauh lebih ringan daripada baseline CNN besar seperti:
- MobileNetV3Large INT8: `8.39 ms`, `5.798 MB`
- EfficientNetLite0 INT8: `14.57 ms`, `5.008 MB`

### 2. INT8 umumnya menguntungkan untuk C8/C10, tapi tidak universal

Contoh:
- `L0.05 C6`: INT8 lebih lambat (`2.05 -> 2.36 ms`)
- `L0.20 C6`: INT8 sedikit lebih cepat (`2.17 -> 2.09 ms`)
- `L0.20 C8`: INT8 jelas lebih cepat (`3.00 -> 2.22 ms`)
- `L0.20 C10`: INT8 juga lebih cepat (`3.77 -> 2.94 ms`)

Artinya, untuk keluarga model terbaru ini, manfaat INT8 tidak bisa diasumsikan sama pada semua kapasitas. Namun untuk kandidat utama C8/C10, hasilnya tetap positif.

### 3. Akurasi INT8 tetap stabil

Untuk model retrain dan KD yang relevan, penurunan INT8 relatif kecil:
- `retrain L0.20 C8`: `98.44% -> 97.96%` (`-0.48 pp`)
- `kd t3 a0.8`: `99.04% -> 98.80%` (`-0.24 pp`)
- `kd t4 a0.8`: `98.92% -> 98.80%` (`-0.12 pp`)

Ini cukup kuat untuk mendukung klaim bahwa model final tetap layak dideploy dalam INT8.

## F. Rekomendasi Narasi Final

Jika hanya ingin satu jalur cerita yang paling rapi:

1. Search `lambda=0.20` menghasilkan baseline `retrain_hwNAS_L0.20_C8_stemds8_834cls`.
2. Baseline ini dipilih karena memberi trade-off yang baik antara akurasi dan efisiensi edge.
3. KD dengan teacher EfficientNetV2-M diuji di atas baseline tersebut.
4. Konfigurasi terbaik `T=3, alpha=0.8` meningkatkan akurasi dari `98.44%` menjadi `99.04%`.
5. Setelah export ONNX dan evaluasi Raspberry Pi, model KD final tetap kecil (`0.596 MB` INT8) dan cepat (`2.27 ms` INT8), dengan akurasi INT8 `98.80%`.

Kalimat inti yang bisa dipakai:

> Pada konfigurasi stem-downsample 8, baseline retrain terbaik untuk jalur hardware-aware NAS + KD dipilih pada model `L0.20 C8`, karena memberikan keseimbangan akurasi-deployment yang baik. Knowledge distillation dari EfficientNetV2-M meningkatkan akurasi test dari 98.44% menjadi 99.04% tanpa perubahan berarti pada latency Raspberry Pi maupun ukuran model INT8, sehingga model `kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed` dipilih sebagai kandidat final.

## G. Kesimpulan Praktis

- Untuk baseline retrain terbaru, `retrain_hwNAS_L0.20_C8_stemds8_834cls` adalah baseline internal paling tepat untuk eksperimen KD.
- Untuk model KD final, `kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed` adalah pilihan paling kuat dan paling rapi untuk dibawa ke laporan utama.
- `retrain_hwNAS_L0.05_C10_stemds8_834cls` tetap penting dicatat sebagai retrain terbaik murni dari sisi akurasi, tetapi kurang ideal dijadikan pasangan baseline-KD karena bukan basis model yang sama dengan rangkaian KD yang diuji.


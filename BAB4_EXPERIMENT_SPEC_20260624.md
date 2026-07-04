# 1. Ringkasan Penelitian

## 1.1 Judul / fokus penelitian

Penelitian ini berfokus pada perancangan model palm vein recognition yang akurat dan efisien untuk deployment edge, dengan pendekatan **hardware-aware NAS**, **knowledge distillation**, dan **post-training quantization (PTQ) INT8**. Target implementasi edge divalidasi menggunakan benchmark inferensi di **Raspberry Pi**.

## 1.2 Tujuan utama penelitian

Tujuan utama penelitian adalah mencari **trade-off terbaik** antara:

- akurasi klasifikasi pada test set 834 kelas,
- efisiensi model (parameter, FLOPs, ukuran model),
- latency inferensi di Raspberry Pi,
- dan stabilitas performa setelah quantization INT8.

Jadi fokus penelitian bukan hanya akurasi maksimum, tetapi **kombinasi akurasi dan efisiensi deploy**.

## 1.3 Kontribusi utama yang ditonjolkan di Bab 4

Kontribusi yang dapat ditonjolkan:

1. Perancangan arsitektur student berbasis **hardware-aware NAS** dengan penalti latency dari lookup table operator yang diukur di perangkat target.
2. Evaluasi sistematis kandidat NAS hasil retrain pada konfigurasi `stem_downsample=8`.
3. Peningkatan performa student melalui **knowledge distillation** dari teacher besar tanpa mengorbankan efisiensi deploy secara berarti.
4. Validasi bahwa model final tetap ringan dan cepat setelah diekspor ke ONNX dan diuji sebagai **FP32 vs INT8** di Raspberry Pi.

---

# 2. Hasil NAS

## 2.1 Ringkasan setup NAS

- Metode NAS: **P-DARTS**
- Search space: 12 operasi
- Resolusi input search: `112 x 112`
- Jumlah kelas: `834`
- Seed: `42`
- Staging P-DARTS:
  - Stage 1: `cells=5`, `epochs=25`, `num_ops=12`
  - Stage 2: `cells=8`, `epochs=25`, `num_ops=7`
  - Stage 3: `cells=11`, `epochs=25`, `num_ops=4`
- Nilai lambda latency-aware yang tersedia:
  - `lambda=0.0`
  - `lambda=0.05`
  - `lambda=0.10`
  - `lambda=0.20`
- Device target untuk latency-aware search:
  - **Raspberry Pi** via ONNX Runtime CPU

Search configuration utama dari `search_summary.json`:

- `C_search = 16`
- `batch_size = 16`
- `alpha_warmup_epochs = 10`
- `search_train_ratio = 0.5`
- `grad_clip = 5.0`
- `label_smoothing = 0.1`

## 2.2 LUT latency yang dipakai di penelitian

Lookup table latency dibangun dengan script [build_latency_lut.py](/Users/fahmitaris/Downloads/NAS-DARTS/build_latency_lut.py:1).

Cara kerjanya:

1. Setiap operator kandidat pada search space diekspor menjadi ONNX kecil untuk beberapa konfigurasi representatif `(C, H, stride)`.
2. Konfigurasi default yang diukur:
   - `(8, 56, 1)`
   - `(16, 28, 1)`
   - `(32, 14, 1)`
   - `(16, 28, 2)`
   - `(32, 14, 2)`
3. ONNX operator kecil ini kemudian diukur latencynya di Raspberry Pi dengan ONNX Runtime CPU.
4. Median latency tiap konfigurasi diambil, lalu dirata-ratakan per operator untuk membentuk LUT final.
5. LUT dipakai sebagai penalti latency pada objective search:
   - `L = L_CE + lambda * expected_latency`
   - `expected_latency` dihitung dari probabilitas operator di edge dikalikan biaya LUT operator tersebut.

File LUT yang tersedia di repo:

- [latency_lut_pi.json](/Users/fahmitaris/Downloads/NAS-DARTS/latency_lut_pi.json)
  - LUT hasil pengukuran latency operator di Raspberry Pi
  - berisi biaya operator dasar hasil pengukuran ONNX per-op
- [latency_lut_pi_int8_corrected.json](/Users/fahmitaris/Downloads/NAS-DARTS/latency_lut_pi_int8_corrected.json)
  - LUT INT8 yang sudah dikoreksi
  - memakai pengurangan `qdq_floor_ms = 0.03299 ms` untuk menghilangkan artifact overhead boundary QDQ pada probe operator terisolasi
  - **ini yang lebih tepat dijadikan LUT utama penelitian**, karena eksperimen hardware-aware diarahkan ke deployment INT8 edge

Operator yang ikut dalam search space:

- `none`
- `skip_connect`
- `sep_conv_3x3`
- `sep_conv_5x5`
- `dil_conv_3x3`
- `dil_conv_5x5`
- `mbconv3_3x3`
- `mbconv6_3x3`
- `rep_conv_3x3`
- `rep_conv_5x5`
- `avg_pool_3x3`
- `max_pool_3x3`

### Nilai LUT Raspberry Pi dasar

| Operator | Cost (ms) |
|---|---:|
| none | 0.01756 |
| skip_connect | 0.02451 |
| sep_conv_3x3 | 0.05361 |
| sep_conv_5x5 | 0.10654 |
| dil_conv_3x3 | 0.04343 |
| dil_conv_5x5 | 0.06384 |
| mbconv3_3x3 | 0.09154 |
| mbconv6_3x3 | 0.15758 |
| rep_conv_3x3 | 0.05515 |
| rep_conv_5x5 | 0.12915 |
| avg_pool_3x3 | 0.02052 |
| max_pool_3x3 | 0.02161 |

### Nilai LUT Raspberry Pi INT8 corrected

| Operator | Cost (ms) |
|---|---:|
| none | 0.00000 |
| skip_connect | 0.00000 |
| sep_conv_3x3 | 0.03860 |
| sep_conv_5x5 | 0.05758 |
| dil_conv_3x3 | 0.02028 |
| dil_conv_5x5 | 0.02886 |
| mbconv3_3x3 | 0.06517 |
| mbconv6_3x3 | 0.08368 |
| rep_conv_3x3 | 0.02414 |
| rep_conv_5x5 | 0.04426 |
| avg_pool_3x3 | 0.01428 |
| max_pool_3x3 | 0.13046 |

### Interpretasi LUT

- Pada LUT dasar maupun LUT INT8 corrected, `mbconv6_3x3` dan `rep_conv_5x5` termasuk operator mahal.
- `rep_conv_3x3` dan `dil_conv_3x3` relatif lebih murah, sehingga wajar jika makin banyak muncul pada lambda yang lebih besar.
- Pada LUT INT8 corrected, `rep_conv_3x3` lebih murah daripada `sep_conv_3x3` dan jauh lebih murah daripada `mbconv6_3x3`.
- `max_pool_3x3` di LUT INT8 corrected tampak mahal. Ini perlu dicatat sebagai hasil pengukuran operator terisolasi, tetapi tidak menjadi operator dominan pada genotype akhir.

Kalimat aman untuk Bab 4:

> Penelitian ini menyiapkan dua versi lookup table latency Raspberry Pi, yaitu LUT pengukuran dasar dan LUT INT8 corrected. Untuk eksperimen hardware-aware yang diarahkan ke deployment INT8, pembahasan utama menggunakan LUT INT8 corrected karena telah mengurangi artifact overhead Quantize/Dequantize pada pengukuran operator terisolasi.

## 2.3 Genotype terbaik per lambda

Sumber:

- `nas_results/search_hwint8_l0.0/search_summary.json`
- `nas_results/search_hwint8_l0.05/search_summary.json`
- `nas_results/search_hwint8_l0.10/search_summary.json`
- `nas_results/search_hwint8_l0.20/search_summary.json`

### Lambda = 0.0

Karakter arsitektur:

- Masih banyak `sep_conv_3x3`
- `rep_conv_3x3` mulai muncul, tetapi belum dominan
- `skip_connect` hanya sedikit

Normal cell:

```text
sep_conv_3x3, sep_conv_3x3, sep_conv_3x3, rep_conv_3x3,
sep_conv_3x3, sep_conv_3x3, skip_connect, sep_conv_3x3
```

Reduce cell:

```text
sep_conv_3x3, sep_conv_3x3, rep_conv_3x3, rep_conv_3x3,
rep_conv_3x3, sep_conv_3x3, rep_conv_3x3, rep_conv_3x3
```

### Lambda = 0.05

Karakter arsitektur:

- `rep_conv_3x3` menjadi dominan
- `dil_conv_3x3` muncul lebih sering
- `skip_connect` lebih aktif
- `sep_conv` hampir hilang

Normal cell:

```text
rep_conv_3x3, rep_conv_3x3, skip_connect, rep_conv_3x3,
dil_conv_3x3, dil_conv_3x3, skip_connect, dil_conv_3x3
```

Reduce cell:

```text
dil_conv_3x3, rep_conv_3x3, rep_conv_3x3, skip_connect,
rep_conv_3x3, rep_conv_3x3, rep_conv_3x3, rep_conv_3x3
```

### Lambda = 0.10

Karakter arsitektur:

- `rep_conv` sangat dominan
- `skip_connect` makin sering
- mulai muncul `rep_conv_5x5`
- arsitektur lebih condong ke pola operator murah

Normal cell:

```text
rep_conv_3x3, rep_conv_3x3, skip_connect, skip_connect,
rep_conv_3x3, rep_conv_3x3, rep_conv_3x3, rep_conv_3x3
```

Reduce cell:

```text
rep_conv_5x5, rep_conv_3x3, rep_conv_5x5, skip_connect,
skip_connect, rep_conv_3x3, rep_conv_3x3, rep_conv_3x3
```

### Lambda = 0.20

Karakter arsitektur:

- `rep_conv_3x3` dominan hampir di seluruh cell
- `skip_connect` tetap ada sebagai jalur murah
- `rep_conv_5x5` hanya muncul terbatas di reduce cell
- ini adalah bentuk arsitektur paling jelas mengarah ke efisiensi operator

Normal cell:

```text
skip_connect, rep_conv_3x3, skip_connect, rep_conv_3x3,
rep_conv_3x3, rep_conv_3x3, rep_conv_3x3, rep_conv_3x3
```

Reduce cell:

```text
skip_connect, rep_conv_3x3, rep_conv_3x3, rep_conv_5x5,
skip_connect, rep_conv_3x3, rep_conv_3x3, rep_conv_3x3
```

## 2.4 Insight utama hasil search

Pola pergeseran antar lambda:

- `lambda=0.0`: lebih bebas memilih operator berbasis separable conv
- `lambda=0.05`: mulai bergeser ke `rep_conv` dan `dil_conv`
- `lambda=0.10`: `rep_conv` semakin dominan
- `lambda=0.20`: arsitektur paling jelas mengutamakan operator murah dan shortcut

Interpretasi:

- semakin besar penalti latency, semakin kuat NAS menghindari operator yang secara empiris lebih mahal di LUT
- hasil search mendukung bahwa objective latency-aware benar-benar memengaruhi bentuk genotype akhir

---

# 3. Hasil Retrain Kandidat

## 3.1 Catatan setup retrain

Semua kandidat retrain yang dibahas di sini memakai:

- `num_cells = 8`
- `stem_downsample = 8`
- `reduction_indices = 2,5`

Pilihan `C6 / C8 / C10` dipakai sebagai sweep kapasitas model:

- `C6`: model paling kecil dan paling cepat
- `C8`: titik tengah kapasitas
- `C10`: model terbesar di keluarga kandidat final

## 3.2 Tabel retrain lengkap

| Model | Lambda | C_init | Best Val Acc | Best Val Acc Epoch | Best Val Loss | Best Epoch (test file) | Test Acc | Params | FLOPs (M) | Model Size (MB) | Local CPU Latency (ms) | Local GPU Latency (ms) | Pi FP32 Mean (ms) | Pi INT8 Mean (ms) | Pi INT8 Acc | Pi INT8 Size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| retrain_hwNAS_L0.05_C6_stemds8_834cls | 0.05 | 6 | 97.00% | 254 | 2.4429 | 209 | 96.52% | 315,754 | 19.9698 | 1.2205 | 7.5652 | 7.2179 | 2.05 | 2.36 | 96.28% | 0.450 |
| retrain_hwNAS_L0.05_C8_stemds8_834cls | 0.05 | 8 | 97.96% | 228 | 2.2949 | 228 | 98.08% | 432,764 | 31.5813 | 1.6715 | 12.8036 | 13.5785 | 2.87 | 2.43 | 98.20% | 0.562 |
| retrain_hwNAS_L0.05_C10_stemds8_834cls | 0.05 | 10 | 99.04% | 226 | 2.2061 | 228 | 99.04% | 573,766 | 45.6703 | 2.2141 | 14.5119 | 10.8594 | 3.52 | 3.17 | 99.28% | 0.694 |
| retrain_hwNAS_L0.20_C6_stemds8_834cls | 0.20 | 6 | 96.64% | 260 | 2.4650 | 296 | 97.72% | 365,722 | 25.2936 | 1.4149 | 7.1891 | 8.8606 | 2.17 | 2.09 | 97.36% | 0.456 |
| retrain_hwNAS_L0.20_C8_stemds8_834cls | 0.20 | 8 | 98.20% | 204 | 2.3239 | 281 | 98.44% | 522,332 | 41.0881 | 2.0182 | 8.5212 | 9.1178 | 3.00 | 2.22 | 97.96% | 0.602 |
| retrain_hwNAS_L0.20_C10_stemds8_834cls | 0.20 | 10 | 99.16% | 224 | 2.2118 | 298 | 98.80% | 714,406 | 60.5643 | 2.7568 | 9.8506 | 9.3002 | 3.77 | 2.94 | 98.92% | 0.782 |

## 3.3 Insight retrain

- Pada kedua keluarga lambda, akurasi meningkat seiring kenaikan kapasitas `C6 -> C8 -> C10`.
- `L0.05 C10` memberi akurasi retrain tertinggi: **99.04%**
- `L0.20 C8` bukan kandidat akurasi terbaik mutlak, tetapi ia berada pada trade-off yang rapi:
  - test accuracy tinggi
  - ukuran model tetap kecil
  - latency Pi tetap rendah
  - kapasitas cukup untuk dijadikan baseline KD

---

# 4. Dasar Pemilihan Kandidat Utama

## 4.1 Kenapa akhirnya memilih `L0.20 C8` sebagai baseline student KD?

Alasan utama:

1. `L0.20 C8` memberi **trade-off terbaik** untuk jalur NAS -> KD -> PTQ.
2. Akurasinya sudah cukup tinggi sebagai baseline (`98.44%`) tanpa membuat model terlalu besar.
3. Di Raspberry Pi, model ini tetap sangat ringan:
   - FP32 mean latency: `3.00 ms`
   - INT8 mean latency: `2.22 ms`
   - INT8 size: `0.602 MB`
4. Dibanding `C6`, model ini punya headroom akurasi yang lebih baik.
5. Dibanding `C10`, model ini lebih representatif sebagai student edge karena lebih kecil dan lebih cepat, sambil mempertahankan akurasi yang sudah kuat.

## 4.2 Kenapa bukan `L0.05`?

`L0.05 C10` memang menang dari sisi akurasi retrain murni (`99.04%`), tetapi untuk narasi metodologi:

- rangkaian eksperimen KD utama yang tersedia dibangun di atas basis `L0.20 C8`
- `L0.20` juga lebih konsisten dengan framing hardware-aware yang lebih kuat
- `L0.20 C8` lebih cocok dijadikan baseline student yang efisien dan representatif untuk deployment edge

Jadi alasan utamanya bukan karena `L0.05` buruk, tetapi karena `L0.20 C8` lebih cocok sebagai **baseline eksperimental KD yang konsisten dan edge-oriented**.

## 4.3 Kenapa bukan `C6`?

- `C6` terlalu kecil dan akurasinya tertinggal jelas dari `C8`
- pada jalur student final, `C6` terlalu agresif jika targetnya ingin menjaga akurasi mendekati 99%

## 4.4 Kenapa bukan `C10`?

- `C10` memang lebih akurat, tetapi:
  - parameter lebih banyak
  - FLOPs lebih tinggi
  - ukuran lebih besar
  - latency Pi lebih lambat
- untuk baseline student KD, `C10` kurang ideal jika ingin mempertahankan karakter lightweight edge model

## 4.5 Argumen utamanya

Argumen utama pemilihan `L0.20 C8`:

- bukan model paling akurat
- bukan model paling cepat
- tetapi model **paling seimbang** dan paling representatif untuk deployment edge dan kelanjutan eksperimen KD

---

# 5. Hasil Teacher

## 5.1 Daftar teacher yang dilatih

Teacher yang tersedia di `Teacher/training_results`:

- ConvNeXtBase
- DenseNet121
- EfficientNetB4
- EfficientNetLite0
- EfficientNetV2M
- InceptionV3
- MobileNetV3Large
- MobileNetV3Small
- RegNetY16GF
- ResNet50
- ShuffleNetV2_x1_0
- VGG16

## 5.2 Metrik teacher

Catatan:

- FLOPs teacher tersedia untuk model yang sudah dihitung via [compute_teacher_flops.py](/Users/fahmitaris/Downloads/NAS-DARTS/compute_teacher_flops.py:1)
- FLOPs belum tersedia di repo untuk `ShuffleNetV2_x1_0` dan `MobileNetV3Small` dari file hasil training, tetapi sudah dihitung terpisah dengan `thop`
- latency Raspberry Pi hanya tersedia untuk subset model yang memang dibenchmark langsung

| Teacher | Test Acc | Params | FLOPs / MMACs | Local Inference per Batch (s) | PTQ Export | Pi Benchmark Available |
|---|---:|---:|---:|---:|---|---|
| ConvNeXtBase | 100.00% | 88,421,314 | 15,373.3 | 0.1636 | Ya | Tidak |
| DenseNet121 | 99.88% | 7,808,706 | 2,896.8 | 0.1476 | Ya | Tidak |
| EfficientNetB4 | 99.76% | 19,043,978 | 1,578.3 | 0.1507 | Ya | Tidak |
| EfficientNetLite0 | 99.88% | 4,439,362 | 367.4 | 0.0197 | Ya | Ya |
| EfficientNetV2M | 100.00% | 53,926,710 | 5,446.3 | 0.1492 | Ya | Tidak |
| InceptionV3 | 99.76% | 26,693,476 | 2,856.2 | 0.1534 | Ya | Tidak |
| MobileNetV3Large | 99.88% | 5,270,386 | 234.6 | 0.1456 | Ya | Ya |
| MobileNetV3Small | 99.64% | 2,372,706 | 62.3 | 0.0251 | Ya | Ya |
| RegNetY16GF | 100.00% | 83,087,990 | 16,008.9 | 0.1688 | Ya | Tidak |
| ResNet50 | 100.00% | 25,216,898 | 4,133.4 | 0.1462 | Ya | Tidak |
| ShuffleNetV2_x1_0 | 99.16% | 2,108,454 | 152.5 | 0.0204 | Ya | Ya |
| VGG16 | 99.64% | 137,677,442 | 15,469.6 | 0.1546 | Ya | Tidak |

## 5.3 Teacher yang dipilih untuk KD

Teacher yang dipakai pada eksperimen KD utama:

- **EfficientNetV2M**

## 5.4 Alasan pemilihannya

Alasan memilih `EfficientNetV2M`:

1. Akurasi teacher sangat tinggi: **100.00%**
2. Model ini berfungsi sebagai teacher kuat untuk menyediakan soft target yang kaya
3. Dibanding teacher ringan seperti MobileNetV3Small atau ShuffleNet, kapasitas representasinya jauh lebih besar
4. Teacher tidak menjadi model deploy utama, sehingga ukuran dan latency besar masih dapat ditoleransi

---

# 6. Hasil KD

## 6.1 Konfigurasi eksperimen utama

Eksperimen utama yang diminta:

- baseline tanpa KD
- `alpha = 1.0`
- `T2 a0.8`
- `T3 a0.8`
- `T4 a0.8`
- `T3 a0.7`
- `T3 a0.9`

Definisi bobot:

- `CE weight = alpha`
- `KD weight = 1 - alpha`

## 6.2 Hasil KD utama

| Konfigurasi | Temperature | Alpha | CE Weight | KD Weight | Best Epoch | Best Val Acc | Best Val Loss | Test Acc | Test Loss | AUC | EER (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline tanpa KD (`retrain_hwNAS_L0.20_C8_stemds8_834cls`) | - | - | 1.0 | 0.0 | 281 | 98.20% | 2.3239 | 98.44% | - | 0.999935 | 0.0065 |
| `alpha=1.0` (`finetune_hwNAS_L0.20_C8_t3_a1.0_ls0_nomix_lr1e4_fixed`) | 3.0 | 1.0 | 1.0 | 0.0 | 12 | 98.32% | 0.0743 | 98.20% | 0.1068 | 0.999983 | 0.0017 |
| `T2 a0.8` | 2.0 | 0.8 | 0.8 | 0.2 | 37 | 98.80% | 0.2555 | 98.92% | 0.2870 | 0.999974 | 0.0026 |
| `T3 a0.8` | 3.0 | 0.8 | 0.8 | 0.2 | 58 | 99.04% | 0.2391 | 99.04% | 0.2564 | 0.999977 | 0.0023 |
| `T4 a0.8` | 4.0 | 0.8 | 0.8 | 0.2 | 37 | 98.80% | 0.2317 | 98.92% | 0.2595 | 0.999980 | 0.0020 |
| `T3 a0.7` | 3.0 | 0.7 | 0.7 | 0.3 | 58 | 98.92% | 0.3122 | 99.04% | 0.3571 | 0.999973 | 0.0027 |
| `T3 a0.9` | 3.0 | 0.9 | 0.9 | 0.1 | 59 | 98.80% | 0.1611 | 98.68% | 0.1633 | 0.999984 | 0.0016 |

## 6.3 Hasil KD tambahan / pendukung

Eksperimen tambahan yang Anda sebut:

- `T3.5 a0.8`
- `T3 a0.75`
- `T3 a0.85`
- `T3 a0.6`

Status saat ini:

- folder hasil eksperimen tersebut **tidak ditemukan** di `knowledge_distilation/kd_results`
- jadi untuk Bab 4, paling aman ditulis sebagai:
  - belum dijalankan, atau
  - tidak dimasukkan ke eksperimen utama

## 6.4 Insight KD utama

- Fine-tuning tanpa KD (`alpha=1.0`) tidak mengungguli baseline retrain.
- Artinya, peningkatan bukan datang dari fine-tuning biasa, tetapi dari informasi teacher.
- Konfigurasi terbaik adalah:
  - `T3 a0.8`
  - `T3 a0.7`
- Keduanya mencapai test accuracy **99.04%**
- Jika butuh satu konfigurasi final yang paling aman untuk dijadikan model utama:
  - **`kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed`**

Alasan memilih `T3 a0.8` sebagai model final lebih kuat:

- akurasi tertinggi tercapai
- bobot CE dan KD seimbang
- loss test lebih rendah dibanding `T3 a0.7`
- konfigurasi lebih stabil secara naratif sebagai kandidat utama

---

# 7. Hasil PTQ

## 7.1 Metode quantization

Metode quantization yang dipakai:

- **Static PTQ**
- format: **QDQ**
- ONNX opset: `13`
- ONNX Runtime quantization
- per-channel weight quantization
- calibration image:
  - root: `/Users/fahmitaris/Downloads/NAS-DARTS-TEMP/preprocessed_results`
  - jumlah umum: `200`

Setting penting:

- preprocessing calibration:
  - grayscale -> 3 channel
  - resize `224 x 224`
  - ImageNet normalization
- graph pre-processing:
  - `quant_pre_process`

## 7.2 Model yang di-PTQ

Model yang jelas punya artefak PTQ di repo:

- teacher models di `Teacher/training_results`
- retrain NAS stem8 yang dibenchmark di Raspberry Pi
- model KD utama yang dibenchmark di Raspberry Pi

## 7.3 Hasil FP32 vs INT8 yang relevan untuk pembahasan

| Model | FP32 Acc | INT8 Acc | Delta Acc (pp) | FP32 Size (MB) | INT8 Size (MB) | FP32 Latency Mean (Pi, ms) | INT8 Latency Mean (Pi, ms) | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV3Large | 99.88% | 98.68% | -1.20 | 21.077 | 5.798 | 15.49 | 8.39 | 1.85x |
| MobileNetV3Small | 99.64% | 99.64% | +0.00 | 9.505 | 2.714 | 6.29 | 5.11 | 1.23x |
| ShuffleNetV2_x1_0 | 99.16% | 99.52% | +0.36 | 8.528 | 2.503 | 6.85 | 5.58 | 1.23x |
| EfficientNetLite0 | 99.88% | 99.76% | -0.12 | 17.780 | 5.008 | 25.01 | 14.57 | 1.72x |
| retrain_hwNAS_L0.20_C8_stemds8_834cls | 98.44% | 97.96% | -0.48 | 1.528 | 0.602 | 3.00 | 2.22 | 1.35x |
| kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed | 99.04% | 98.80% | -0.24 | 1.526 | 0.596 | 3.00 | 2.27 | 1.32x |

## 7.4 Insight PTQ

- Pada model NAS/KD final, PTQ menurunkan ukuran model secara konsisten dan mempercepat inferensi.
- Penurunan akurasi setelah PTQ relatif kecil:
  - baseline retrain utama: `-0.48 pp`
  - KD final utama: `-0.24 pp`
- Jadi PTQ tetap layak dimasukkan sebagai tahap akhir deployment.

Catatan khusus:

- `MobileNetV3Small` di repo saat ini memakai recipe INT8 yang sudah diperbaiki, sehingga hasil INT8 tetap stabil.

---

# 8. Hasil Benchmark Raspberry Pi

## 8.1 Model yang wajib ditampilkan

Untuk Bab 4, minimal model berikut memang layak ditampilkan:

- MobileNetV3Large
- MobileNetV3Small
- ShuffleNetV2_x1_0
- EfficientNetLite0
- `retrain_hwNAS_L0.20_C8_stemds8_834cls`
- `kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed`

## 8.2 Tabel benchmark Raspberry Pi

| Model | FP32 Acc | FP32 Size (MB) | FP32 Mean | FP32 Median | FP32 p95 | INT8 Acc | INT8 Size (MB) | INT8 Mean | INT8 Median | INT8 p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV3Large | 99.88% | 21.077 | 15.49 | 14.77 | 19.87 | 98.68% | 5.798 | 8.39 | 8.05 | 9.80 |
| MobileNetV3Small | 99.64% | 9.505 | 6.29 | 6.01 | 7.23 | 99.64% | 2.714 | 5.11 | 4.89 | 5.83 |
| ShuffleNetV2_x1_0 | 99.16% | 8.528 | 6.85 | 6.57 | 7.69 | 99.52% | 2.503 | 5.58 | 5.39 | 6.31 |
| EfficientNetLite0 | 99.88% | 17.780 | 25.01 | 23.67 | 34.38 | 99.76% | 5.008 | 14.57 | 13.93 | 20.34 |
| retrain_hwNAS_L0.20_C8_stemds8_834cls | 98.44% | 1.528 | 3.00 | 2.88 | 3.50 | 97.96% | 0.602 | 2.22 | 2.14 | 2.54 |
| kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed | 99.04% | 1.526 | 3.00 | 2.88 | 3.63 | 98.80% | 0.596 | 2.27 | 2.20 | 2.51 |

## 8.3 Model pembanding lain yang juga tersedia di Raspberry Pi

Jika ingin menambah pembanding internal NAS:

| Model | FP32 Acc | INT8 Acc | FP32 Mean (ms) | INT8 Mean (ms) |
|---|---:|---:|---:|---:|
| retrain_hwNAS_L0.05_C6_stemds8_834cls | 96.52% | 96.28% | 2.05 | 2.36 |
| retrain_hwNAS_L0.05_C8_stemds8_834cls | 98.08% | 98.20% | 2.87 | 2.43 |
| retrain_hwNAS_L0.05_C10_stemds8_834cls | 99.04% | 99.28% | 3.52 | 3.17 |
| retrain_hwNAS_L0.20_C6_stemds8_834cls | 97.60% | 97.36% | 2.17 | 2.09 |
| retrain_hwNAS_L0.20_C10_stemds8_834cls | 98.80% | 98.92% | 3.77 | 2.94 |
| kd_hwNAS_L0.20_C8_t2_a0.8_ls0_nomix_lr1e4_fixed | 98.92% | 98.68% | 2.98 | 2.30 |
| kd_hwNAS_L0.20_C8_t3_a0.7_ls0_nomix_lr1e4_fixed | 99.04% | 98.44% | 2.95 | 2.29 |
| kd_hwNAS_L0.20_C8_t3_a0.9_ls0_nomix_lr1e4_fixed | 98.68% | 98.32% | 2.95 | 2.26 |
| kd_hwNAS_L0.20_C8_t4_a0.8_ls0_nomix_lr1e4_fixed | 98.92% | 98.80% | 2.99 | 2.33 |
| finetune_hwNAS_L0.20_C8_t3_a1.0_ls0_nomix_lr1e4_fixed | 98.20% | 98.20% | 2.97 | 2.28 |

---

# 9. Hasil Pembanding Eksternal

## 9.1 Tabel ringkas pembanding utama

| Model | Role | Test Acc | Params | FLOPs (M) | Size FP32 (MB) | Pi FP32 Mean (ms) |
|---|---|---:|---:|---:|---:|---:|
| EfficientNetV2M | Teacher utama | 100.00% | 53,926,710 | 5,446.3 | 215.348 | - |
| MobileNetV3Small | Baseline lightweight | 99.64% | 2,372,706 | 62.3 | 9.505 | 6.29 |
| retrain_hwNAS_L0.20_C8_stemds8_834cls | NAS baseline utama | 98.44% | 522,332 | 41.0881 | 1.528 | 3.00 |
| kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed | NAS + KD final | 99.04% | sekitar 522k backbone yang sama | sekitar 41.1 | 1.526 | 3.00 |

Interpretasi:

- model NAS final tetap jauh lebih kecil daripada teacher
- model NAS final juga jauh lebih cepat daripada baseline lightweight CNN besar di Raspberry Pi
- KD mendorong model NAS mendekati akurasi baseline ringan/teacher dengan biaya deploy yang sangat kecil

---

# 10. Keputusan Akhir yang Ingin Diambil di Bab 4

## 10.1 Model final utama tesis

Model final utama yang paling layak ditonjolkan:

- **`kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed`**

Alasan:

- test accuracy tertinggi di jalur KD utama: `99.04%`
- baseline student yang jelas
- hasil deploy ONNX dan INT8 tetap kuat
- latency Pi tetap rendah

## 10.2 Model pembanding utama

Model pembanding utama yang sebaiknya dipakai:

- baseline student retrain:
  - `retrain_hwNAS_L0.20_C8_stemds8_834cls`
- baseline lightweight CNN:
  - `MobileNetV3Small`
  - `ShuffleNetV2_x1_0`
  - `EfficientNetLite0`
- baseline teacher / accuracy ceiling:
  - `EfficientNetV2M`

## 10.3 Hal yang paling ingin ditonjolkan

Yang paling kuat untuk ditonjolkan:

- **model final setelah KD + PTQ**
- dengan framing:
  - akurasi tinggi,
  - ukuran sangat kecil,
  - latency Raspberry Pi rendah,
  - dan ada pembuktian bahwa KD menaikkan akurasi baseline retrain tanpa menghilangkan karakter edge-efficient

Kalau ingin secondary headline:

- model tercepat:
  - `retrain_hwNAS_L0.05_C6_stemds8_834cls`
- model paling seimbang:
  - `kd_hwNAS_L0.20_C8_t3_a0.8_ls0_nomix_lr1e4_fixed`

---

# 11. Catatan Keputusan / Insight Penting

1. `L0.05 C10` adalah retrain murni dengan akurasi tertinggi, tetapi bukan baseline paling cocok untuk rangkaian KD utama.
2. `L0.20 C8` adalah baseline student paling representatif untuk deployment edge.
3. KD benar-benar memberi manfaat; fine-tuning tanpa KD tidak cukup.
4. `T3 a0.8` adalah konfigurasi KD final yang paling kuat dan paling mudah dipertahankan secara naratif.
5. PTQ INT8 layak dipakai pada model final karena penurunan akurasi kecil dan keuntungan ukuran/latency nyata.
6. Untuk Bab 4, sebaiknya jangan memecah fokus terlalu jauh ke semua konfigurasi tambahan yang belum ada hasilnya.
7. Eksperimen KD tambahan seperti `T3.5 a0.8`, `T3 a0.75`, `T3 a0.85`, dan `T3 a0.6` belum tersedia di repo saat ini, jadi jangan diposisikan sebagai hasil utama.

# NAS-DARTS Palm Vein — Ringkasan Temuan (untuk paper)

Dokumen ini merangkum temuan terverifikasi agar tidak tercecer. Semua latency diukur di
**Raspberry Pi** (onnxruntime CPU, 4 threads); akurasi pada **test set 834 kelas**.
Semua INT8 = static PTQ **per-channel** (QDQ, opset 13). Konfigurasi spasial dikunci:
`stem_downsample=4`, `reduction_indices=2,5`.

---

## Kontribusi inti (3 sumbu terpisah, masing-masing dengan bukti)

1. **Spatial schedule = lever latency dominan, hampir accuracy-neutral.**
   mbconv C4: `stem=2` → 20.46 ms / 98.08% ; `stem=4` → 4.69 ms / 97.24% (**~4.4× lebih cepat**, akurasi ~setara). Param/genotype identik — hanya jadwal reduksi spasial berubah.

2. **Operator = penentu quantization-friendliness, via ARITHMETIC INTENSITY** (bukan FLOPs/param, bukan sekadar jumlah node QDQ). Lihat bagian "Mekanisme INT8". Temuan tambahan pada `mobile_v2_C3`: INT8 juga lebih lambat dari FP32 meskipun topologi MBConv/mobile, sehingga isu latency bukan spesifik hwNAS saja.

3. **Presisi deploy = keputusan PER-MODEL** (FP32 untuk sel compact, INT8 untuk sel padat/reparameterized). Dipilih dengan mengukur keduanya di perangkat.

---

## Tabel Pareto final (Pi, iso-config stem=4)

### FP32
| Model | Op | C | Params | Akurasi | Size | Latency |
|---|---|---|---|---|---|---|
| **hwNAS λ0.05 C6** | rep+dil+skip | 6 | 315k | 97.96% | 0.79 MB | **3.94 ms** ← tercepat |
| mbconv_C4 | mbconv | 4 | 239k | 97.24% | 0.56 MB | 4.69 ms |
| hwNAS λ0.05 C8 | rep+dil+skip | 8 | 433k | 98.08% | 1.21 MB | 5.72 ms |
| repconv_C8_mid14 | rep_conv | 8 | 503k | 98.80% | 1.46 MB | 6.20 ms |
| mbconv_C6 | mbconv | 6 | 338k | 99.28% | 0.94 MB | 7.16 ms |
| mbconv_C8 | mbconv | 8 | 461k | 99.40% | 1.40 MB | 10.07 ms |
| MobileNetV3Large | — | — | ~5.4M | 99.88% | 21.08 MB | ~15.5 ms |

### INT8 (per-channel)
| Model | Akurasi | Size | Latency |
|---|---|---|---|
| hwNAS λ0.05 C6 | 97.96% | 0.45 MB | 5.06 ms |
| hwNAS λ0.05 C8 | 98.32% | 0.56 MB | 5.44 ms |
| repconv_C8_mid14 | 98.92% | 0.60 MB | **5.47 ms** |
| mbconv_C4 | 97.72% | 0.60 MB | 5.80 ms |
| mbconv_C8 | 99.28% | 0.87 MB | 8.36 ms |
| MobileNetV3Large (fair) | 98.68% | 5.80 MB | 8.44 ms |
| MobileNetV3Large (per-tensor, BUG) | 81.06% | 5.54 MB | 8.32 ms |

**Rekomendasi deploy:** hwNAS_C6 **FP32** (speed-champion, 3.94 ms) ; repconv_C8_mid14 **INT8** (5.47 ms) ; mbconv_C6/C8 untuk kebutuhan akurasi tertinggi.

---

## Mekanisme INT8 (terbukti: struktur node + timing + kontras)

**Klaim:** Manfaat PTQ INT8 di CPU edge = fungsi **arithmetic intensity konvolusi**, bukan FLOPs/param maupun jumlah node QDQ.

### Bukti struktural (platform-independent)
Graph INT8 hwNAS_C6: 655 node, di antaranya **488 QuantizeLinear/DequantizeLinear (75%)**; jumlah Conv sama dengan FP32 (93). Graph INT8 = 2.7× node FP32.

### Bukti timing + kontras (Pi, profile_onnx_operators.py)
| | C6 compact (skip/dil) | repconv_C8 (conv padat) |
|---|---|---|
| conv FP32 (summed kernel) | 5.32 ms | 7.01 ms |
| conv INT8 (QLinearConv) | 4.80 ms | 4.12 ms |
| **hemat compute int8** | **0.52 ms (10%)** | **2.89 ms (41%)** |
| overhead konversi (Transpose+Q+DQ) | ~2.4 ms | ~2.7 ms (≈ sama) |
| end-to-end FP32→INT8 | 4.08 → **5.07 (RUGI +24%)** | 6.29 → **5.59 (UNTUNG −11%)** |

**Aturan:** INT8 menguntungkan ⟺ (hemat compute conv) > (overhead konversi tetap).
Overhead konversi hampir konstan (~2.4–2.7 ms); yang menentukan adalah besar hemat compute,
yang ditentukan arithmetic intensity: conv padat/reparameterized (intensity tinggi) → hemat
besar → menang; sel compact memory-bound (skip/dilated, intensity rendah) → hemat kecil → kalah.

### Bukti tambahan: mobile_v2_C3 (MBConv/topologi mobile)
Benchmark Raspberry Pi 5, ONNX Runtime CPU 4 threads, test 834 image:

| Model | Akurasi | Size | Mean latency | Median | p95 |
|---|---:|---:|---:|---:|---:|
| mobile_v2_C3 FP32 | 94.12% (785/834) | 0.408 MB | **13.82 ms** | 12.93 ms | 18.29 ms |
| mobile_v2_C3 INT8 static | 93.05% (776/834) | 0.537 MB | **14.36 ms** | 13.81 ms | 17.07 ms |

Delta INT8 vs FP32: latency **0.96×** (lebih lambat) dan akurasi **−1.08 pp**. Catatan: ukuran file pada run ini 0.537 vs 0.408 MB, tetapi ini **tidak dipakai sebagai klaim utama** karena seri hwNAS final justru menunjukkan ukuran INT8 konsisten lebih kecil.

Interpretasi:
- Ini memperkuat bahwa "INT8 selalu lebih cepat" tidak benar pada CPU edge.
- Pada model mobile kecil, penghematan compute dari INT8 bisa kalah oleh overhead format/runtime quantized: node Q/DQ, scale/zero-point, layout/memory movement, dan fragmentasi graph.
- Karena topologi `mobile_v2_C3` berbasis MBConv/depthwise/pointwise juga mengalami perlambatan, penyebabnya bukan hanya genotype hwNAS, melainkan interaksi **operator mobile + channel kecil + arithmetic intensity rendah + overhead runtime/QDQ**.

### Bukti utama seri hwNAS: size selalu turun, latency tergantung C
Benchmark Raspberry Pi 5, ONNX Runtime CPU 4 threads, test 834 image:

| Model | FP32 acc/size/lat | INT8 acc/size/lat | Speedup | Size gain | Δacc |
|---|---|---|---:|---:|---:|
| hwNAS λ0.05 C4 | 93.29% / 0.457 MB / 2.53 ms | 93.29% / 0.361 MB / 3.75 ms | 0.67× | 1.27× | +0.00 pp |
| hwNAS λ0.05 C6 | 97.96% / 0.790 MB / 3.99 ms | 97.96% / 0.450 MB / 5.10 ms | 0.78× | 1.76× | +0.00 pp |
| hwNAS λ0.20 C4 | 91.49% / 0.530 MB / 2.64 ms | 89.81% / 0.340 MB / 3.36 ms | 0.79× | 1.56× | −1.68 pp |
| hwNAS λ0.20 C6 | 97.60% / 0.963 MB / 4.27 ms | 97.36% / 0.459 MB / 4.47 ms | 0.95× | 2.10× | −0.24 pp |
| hwNAS λ0.10 C8 | 99.16% / 1.621 MB / 6.75 ms | 99.04% / 0.645 MB / 5.70 ms | 1.18× | 2.51× | −0.12 pp |
| hwNAS λ0.20 C8 | 99.16% / 1.527 MB / 6.29 ms | 98.92% / 0.605 MB / 5.27 ms | 1.19× | 2.52× | −0.24 pp |

Pola utama:
- **Ukuran:** pada seri hwNAS final, INT8 **selalu lebih kecil** (1.27× sampai 2.52×).
- **Latency:** INT8 memperlambat C4/C6, tetapi mempercepat C8. Titik balik empiris ada di sekitar C8.
- **Akurasi:** C8 hanya turun kecil (−0.12 sampai −0.24 pp); C4 λ0.20 turun lebih besar (−1.68 pp), sehingga C4 tidak layak jadi model final.

---

## Hardware-aware NAS (search + sweep λ)

- Search space 12 ops (`nas_config.PRIMITIVES`), P-DARTS 3 stage, penalti latency
  `L = L_CE + λ·Σ softmax(α)·LUT[op]`, LUT diukur di Pi.
- **Pergeseran operator vs λ** (bukti kualitatif, dari genotype):
  - λ=0.0 → sep_conv dominan (akurasi, mahal di LUT)
  - λ=0.05 → rep_conv + dil_conv + skip (latency-aware)
  - (λ=0.10/0.20 pending)
- **hwNAS λ0.05 C6 (FP32) mendominasi mbconv_C4**: 97.96% > 97.24% DAN 3.94 < 4.69 ms.
  → NAS sadar-hardware menemukan titik Pareto kecepatan-ekstrem yang mengungguli desain manual.

### Antisipasi pertanyaan reviewer: "pakai LUT INT8 tapi FP32 yang terbaik?"
- Ranking operator **precision-robust**: Spearman ρ = **0.83** (conv-only) antara LUT FP32 dan INT8-terkoreksi → arsitektur hasil search praktis sama apapun presisi LUT. Posisikan LUT sebagai **"device operator-affinity"**, presisi deploy dipilih terpisah per-model.
- "INT8 tak selalu optimal untuk sel compact" = **temuan**, bukan cacat.

---

## Catatan jujur / batasan (untuk ditulis di paper)

1. **Single seed (42).** Klaim head-to-head (mis. hwNAS_C6 vs mbconv_C4, selisih ~0.7%) perlu **≥3 seed + uji McNemar** sebelum diklaim signifikan.
2. **Bias aditivitas LUT INT8** (probe terisolasi membayar QDQ penuh) → dikoreksi via pengurangan floor; tetap perlu **validasi end-to-end** (LUT-prediksi vs latency Pi terukur).
3. **NAS bukan juara akurasi.** hwNAS (97.96–98.32%) ~1% di bawah mbconv_C6/C8 (99.3–99.4%). NAS mengisi pojok kecepatan, bukan akurasi. Jangan over-claim.
4. **QARepVGG (lit.)** menyatakan reparam-conv jelek di INT8; hasil repconv_C8 kita justru baik (98.92%) — **bahas eksplisit** (kemungkinan karena per-channel + fusi conv padat).

---

## To-do tersisa
- [ ] λ=0.10 / 0.20 selesai → dedupe genotype → screening C8 untuk yang unik.
- [ ] (opsional bulletproof) re-run search λ=0.05 dengan LUT FP32 → tunjukkan genotype identik.
- [ ] KD pada **hwNAS_C6** (headroom optimisasi sah) untuk angkat akurasi pada latency tetap.
- [ ] Model final → **3 seed** + McNemar + validasi LUT-vs-terukur.
- [ ] Gambar **Pareto plot** (akurasi vs latency Pi) + **tabel ablasi kuantisasi** (per-tensor vs per-channel) + **tabel profiling QDQ** (C6 vs repconv_C8).


---

## Framing Metodologi: NAS + KD + Quantization (ketiganya tetap pilar)

Quantization **tetap** dalam metodologi — perannya: **dimensi deployment dengan aturan pemilihan presisi per-model**, bukan "kompresi yang selalu menurunkan latency".

**Manfaat kuantisasi terpisah jadi tiga sumbu:**
- **Ukuran model: konsisten untung pada seri hwNAS final.** INT8 mengecilkan model 1.27×–2.52×: C4 λ0.05 0.457→0.361 MB; C6 λ0.05 0.790→0.450 MB; C8 λ0.20 1.527→0.605 MB. Jadi untuk model headline hwNAS, manfaat storage jelas. Catatan terpisah: ukuran ONNX tetap perlu dilaporkan per-model karena overhead QDQ/metadata bisa berbeda antar eksport/protokol.
- **Latency: kondisional** (fungsi arithmetic intensity). Untung pada model compute-bound/padat (C8, repconv, MobileNet); rugi pada sel compact/memory-bound (C4/C6 tertentu, mobile_v2_C3) karena overhead konversi QDQ + layout/memory movement > hemat compute.
- **Akurasi: umumnya terjaga**, tetapi tetap perlu divalidasi per-model. Banyak model int8 ≈ fp32, tetapi `mobile_v2_C3` turun 94.12%→93.05% (−1.08 pp), sehingga klaim aman adalah "PTQ tidak selalu merusak akurasi, tetapi dampaknya model-dependent".

**Aturan deployment per-model (deliverable):**
> Pilih presisi yang meminimalkan latency dengan akurasi ≥ ambang & ukuran ≤ budget:
> - sel compact memory-bound / C kecil → **FP32** bila target utama latency; INT8 tetap berguna bila target utama storage.
> - sel padat/reparameterized / C besar → **INT8** (latency & ukuran biasanya sama-sama untung).

**Posisi kontribusi quantization:** bukan "PTQ selalu menolong" (klaim rapuh), melainkan **karakterisasi kapan PTQ menolong/merugikan untuk edge palm-vein, dengan mekanisme (QDQ-overhead vs compute-saving / arithmetic intensity) dan bukti (struktur node + profiling Pi)**. Ini negative-but-explained result yang dihargai reviewer.

**Catatan:** QAT TIDAK relevan untuk masalah ini — akurasi int8 sudah baik; masalahnya latency yang bersifat struktural (node QDQ/Transpose), yang tidak dihapus oleh QAT.


---

## Pareto FINAL terukur di Pi (semua model, presisi deploy terbaik)

| Model | FP32 acc/lat | INT8 acc/lat | Deploy | Latency | Size |
|---|---|---|---|---|---|
| hwNAS l0.05 C6 | 97.96% / 3.99 | 97.96% / 5.10 | **FP32** | **3.99 ms** | 0.79 MB |
| hwNAS l0.20 C6 | 97.60% / 4.27 | 97.36% / 4.47 | FP32 | 4.27 ms | 0.96 MB |
| hwNAS l0.05 C8 | 98.08% / 5.81 | 98.32% / 5.46 | INT8 | 5.46 ms | 0.56 MB |
| **hwNAS l0.20 C8** | 99.16% / 6.29 | **98.92% / 5.27** | **INT8** | **5.27 ms** | 0.61 MB |
| hwNAS l0.10 C8 | 99.16% / 6.75 | 99.04% / 5.70 | INT8 | 5.70 ms | 0.65 MB |
| repconv_C8_mid14 (manual) | 98.80% / 6.20 | 98.92% / 5.47 | INT8 | 5.47 ms | 0.60 MB |
| mbconv_C6 | 99.28% / 7.16 | 99.28% / 8.14 | FP32 | 7.16 ms | 0.94 MB |
| mbconv_C8 | 99.40% / 10.07 | 99.28% / 8.36 | INT8 | 8.36 ms | 0.87 MB |
| MobileNetV3L | 99.88% / 15.49 | 98.68% / 8.39 | FP32 | 15.49 ms | 21 MB |

(hwNAS C4 dibuang: l0.05 C4 93.29%, l0.20 C4 91.49% — kapasitas terlalu kecil.)

### Frontier Pareto (non-dominated)
- **l0.05 C6 FP32** — 97.96% @ **3.99 ms** (tercepat acceptable)
- **l0.20 C8 INT8** — 98.92% @ **5.27 ms**, 0.61 MB
- **l0.10 C8 INT8** — 99.04% @ 5.70 ms
- mbconv_C6 FP32 — 99.28% @ 7.16 ms
- MobileNetV3L — 99.88% @ 15.49 ms (plafon akurasi, 21 MB)

### Dua kemenangan NAS (dominasi terukur)
1. **l0.20 C8 INT8 (98.92%@5.27ms) mendominasi repconv_C8_mid14 manual (98.92%@5.47ms)** — akurasi sama, lebih cepat + lebih kecil. NAS sadar-hardware mengalahkan substitusi rep manual.
2. **mbconv_C6 (99.28%@7.16ms) mendominasi mbconv_C8-int8 & MobileNet-int8** (akurasi ≥, lebih cepat/kecil).

### Konfirmasi mekanisme INT8 lintas keluarga (arithmetic intensity)
INT8 mempercepat HANYA di C8 (l0.05/0.10/0.20 C8: 1.06–1.19×); memperlambat di C4/C6 (0.67–0.95×). Titik-balik ~C8. Konsisten di 7 model → bukti kuat aturan "INT8 untung ⟺ hemat compute conv > overhead konversi".

Tambahan validasi lintas-topologi: `mobile_v2_C3` juga menunjukkan FP32 lebih cepat daripada INT8 (13.82 ms vs 14.36 ms, 0.96×), dan INT8 menurunkan akurasi 1.08 pp. Ini menunjukkan pola bukan hanya akibat topologi hwNAS, tetapi berlaku pada model mobile/MBConv kecil ketika arithmetic intensity rendah dan overhead quantization/runtime lebih dominan.

## Referensi pendukung mekanisme

Referensi yang mendukung framing akademik:
- Ma et al., **ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design**, ECCV 2018. FLOPs bukan proxy tunggal untuk speed; latency dipengaruhi memory access cost, platform, fragmentasi operator, dan element-wise operation.
- Wang et al., **HAQ: Hardware-Aware Automated Quantization with Mixed Precision**, CVPR 2019. Kebijakan quantization harus hardware-aware; fixed quantization tidak optimal untuk semua arsitektur/layer/hardware.
- Zhang et al., **High Performance Depthwise and Pointwise Convolutions on Mobile Devices**, AAAI 2020. Depthwise/pointwise convolution pada ARM dapat dibatasi cache miss, poor data reuse, dan multicore scalability.
- Sze et al., **Efficient Processing of Deep Neural Networks: A Tutorial and Survey**, Proceedings of the IEEE 2017. Efisiensi DNN dipengaruhi data movement, memory hierarchy, dan reuse, bukan hanya jumlah MAC/FLOPs.
- Tan et al., **MnasNet: Platform-Aware Neural Architecture Search for Mobile**, CVPR 2019. Latency perlu diukur langsung pada hardware target karena proxy seperti FLOPs/parameter tidak cukup.

## Pemilihan pemenang (berbasis skenario deploy)
- **Skenario A (akurasi-tinggi, rekomendasi headline):** **l0.20 C8 INT8** — 98.92%, 5.27 ms, 0.61 MB. Tercepat di kelas ~99%, dominasi repconv manual.
- **Skenario B (tercepat):** **l0.05 C6 FP32** — 97.96%, 3.99 ms, 0.79 MB.
- **Skenario C (akurasi maks):** mbconv_C6 FP32 — 99.28%, 7.16 ms.

## Framing Evaluasi Baseline

Untuk paper/thesis, baseline perlu dipisah menjadi dua kelompok agar klaim edge tidak kabur:

### A. Deployment / lightweight baselines (tabel utama Raspberry Pi 5)
Kelompok ini wajib dievaluasi di Raspberry Pi 5 karena relevan langsung dengan klaim efisiensi edge:
- **Proposed NAS:** hwNAS l0.05 C6 FP32; hwNAS l0.20 C8 INT8.
- **Lightweight baselines:** MobileNetV3Small; ShuffleNetV2_x1_0; EfficientNetLite0.
- **Existing baseline lama:** MobileNetV3Large.

Metrik wajib: akurasi, parameter, size ONNX, latency FP32, latency INT8, mean/median/p95 latency, dan presisi deploy terbaik per-model.

### B. Large CNN / teacher candidates (tabel pendukung)
Kelompok ini dipakai sebagai **accuracy ceiling** dan kandidat teacher, bukan sebagai target deploy utama:
- EfficientNetV2M, EfficientNetB4, ConvNeXtBase, RegNetY16GF, DenseNet121, ResNet50, VGG16, InceptionV3.

Metrik utama cukup: akurasi, parameter, training/test result, dan ukuran model. Benchmark Raspberry Pi 5 untuk semua model besar **tidak wajib** karena mereka tidak dirancang sebagai edge deployment baseline. Jika waktu memungkinkan, benchmark 1–2 model besar saja (mis. EfficientNetV2M dan ConvNeXtBase) untuk menunjukkan trade-off: akurasi tinggi tetapi tidak edge-feasible.

Kalimat framing:
> Large CNNs are evaluated as accuracy upper bounds and teacher candidates, while Raspberry Pi deployment benchmarking focuses on lightweight baselines and the proposed NAS models. This separation avoids conflating accuracy-ceiling models with edge-deployable models.

## Status KD (hwNAS_C6, λ=0.05)
| config | test_acc |
|---|---|
| pre-KD | **97.96%** |
| t4_a0.3 | 97.60% (flat/turun) |
| t8_a0.3 | 97.72% (flat/turun) |
| t4_a0.1, t8_a0.1 | (jalan) |

Sejauh ini **KD flat/marginal** — konsisten dengan task separabilitas tinggi (headroom tipis) + sebagian gap bersifat kapasitas (tak bisa ditambal KD). Jika α=0.1 juga flat → laporkan jujur: "KD marginal; manfaat pipeline dari NAS+quant". Pilih config by VAL, bukan test. Jangan p-hack T/α.

## To-do (update)
- [ ] Selesaikan KD α=0.1 (C6); jika ada gain → apply config terbaik ke l0.20 C8; jika flat → tutup KD sebagai temuan jujur.
- [ ] **3 seed + McNemar** pada model headline (l0.20 C8) vs repconv_C8_mid14 & mbconv_C6 (validasi dominasi single-seed).
- [ ] Plot Pareto akurasi-vs-latency Pi (figure utama).

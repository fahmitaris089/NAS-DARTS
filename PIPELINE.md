# NAS-DARTS Palm Vein — Pipeline Runbook

End-to-end urutan menjalankan script: **Search → Retrain → Export ONNX → INT8 Quantize → KD → Benchmark (FP32 & INT8)**.
Dibuat agar perintah tidak lupa dan konsisten. Semua flag di sini sudah dicocokkan dengan `argparse` masing-masing script.

> ⚠️ **PELAJARAN PENTING (jangan diulang):** baseline INT8 yang adil **wajib** per-channel.
> ONNX harus di-export **opset ≥ 13**. ONNX opset 12 menyebabkan `quantize_static` jatuh ke
> **per-tensor** dan akurasi MobileNetV3 anjlok 99.88% → 81.06% (palsu). Setelah diperbaiki
> (opset 13 + per-channel + pre-process) akurasi pulih ke 98.68%. Lihat bagian
> [Catatan Kuantisasi](#catatan-kuantisasi--pitfall) di bawah.

---

## 0. Konvensi path dataset

`data_dir` berbeda per mesin — sesuaikan:

| Mesin | data_dir | split |
|---|---|---|
| Server (training) | `/workspace/preprocessed_results` | `split_info.json` |
| Mac (lokal) | `/Users/fahmitaris/Downloads/NAS-DARTS-TEMP/preprocessed_results` | `split_info.json` |
| Raspberry Pi (deploy) | `preprocessed_results` | `split_info.json` |

Label map = subject ID di-*sort numerik* → index (`build_label_map`). Sama di semua script.

---

## 1. Search (P-DARTS) — `search.py`

Mencari genotype (topologi sel). Output: `<output_dir>/genotype_final.json`.

```bash
python search.py \
    --data_dir /workspace/preprocessed_results \
    --split_path split_info.json \
    --output_dir nas_results/search_mobile_v2 \
    --batch_size 64 \
    --search_input_size 96 \
    --seed 42
```

Opsional (hardware-aware search dengan LUT latency Pi):
```bash
    --oplat_lambda 0.05 \
    --latency_lut latency_lut_pi.json
```
- `--oplat_lambda 0` = DARTS murni (default). 0.01–0.2 = penalti latency per-operator.
- LUT dibuat oleh `build_latency_lut.py` (diukur di Pi).

---

## 2. Retrain — `retrain.py`

Melatih ulang arsitektur terpilih dari genotype. `--genotype` **wajib**.
Output: `best_model.pth`, `config.json`, `test_results.json`, kurva, dll.

**Konfigurasi pemenang saat ini (`repconv_C8_mid14`): C8 + reduksi spasial.**
```bash
python retrain.py \
    --genotype nas_results/search_mobile_v2/genotype_final.json \
    --data_dir /workspace/preprocessed_results \
    --split_path split_info.json \
    --output_dir nas_results/retrain_repconv_C8_mid14_834cls \
    --C_init 8 \
    --num_cells 8 \
    --stem_downsample 4 \
    --reduction_indices 2,5 \
    --epochs 300 \
    --batch_size 64 \
    --lr 0.001 \
    --weight_decay 0.05 \
    --drop_path_prob 0.2 \
    --cutout_length 16 \
    --augmentation_policy v1_legacy \
    --seed 42
```

Flag kunci efisiensi latency (lever dominan di Pi):
- `--stem_downsample` : 2 = 224→112 (default), **4 = 224→56 (latency jauh lebih rendah)**.
- `--reduction_indices` : posisi reduction cell, mis. `2,5`. Default `[n//3, 2n//3]`.
- `--C_init` : lebar channel awal (4/8/14...). C4 ≈ 0.24M params, C8 ≈ 0.50M params.

> Catatan: `stem_downsample` + `reduction_indices` mengubah resolusi spasial **tanpa**
> mengubah jumlah parameter, tetapi memangkas latency Pi drastis (25→6 ms pada C8).

---

## 3. Export ONNX (model NAS) — `export_retrain_run6_plus2_onnx.py`

Membaca `config.json` + `best_model.pth` di `--model-dir`, menulis `model_benchmark.onnx`
(+ `model_benchmark_metadata.json`). **Default opset 13 — jangan turunkan.**

> 🔴 **WAJIB pakai `export_retrain_run6_plus2_onnx.py` untuk model RETRAIN.**
> JANGAN pakai `export_kd_onnx.py` (itu khusus model hasil KD). `export_kd_onnx.py`
> **tidak membaca `stem_downsample`/`reduction_indices`** dari config retrain, sehingga
> membangun arsitektur stem_downsample=2 lalu memuat bobot stem_downsample=4 →
> **forward salah → akurasi ONNX = 0.12% (1/834, acak)** meski training 97%+.
> Hanya `export_retrain_run6_plus2_onnx.py` yang merekonstruksi stem/reduction dengan benar.
>
> **Gejala salah exporter:** akurasi ONNX ≈ 0.12% sementara `test_results.json` tinggi,
> DAN `model_benchmark_metadata.json` punya field `kd_config` (tanda di-export oleh KD exporter).

```bash
python export_retrain_run6_plus2_onnx.py \
    --model-dir nas_results/retrain_repconv_C8_mid14_834cls \
    --opset 13
```
- `--include-embeddings` (default) export logits + embedding 128-d. `--logits-only` untuk logits saja.

---

## 4. INT8 Static Quantization + latency desktop — `benchmark_int8_static.py`

Kuantisasi **per-channel** (sudah di-hardening: auto opset≥13, `quant_pre_process`,
**tanpa fallback per-tensor senyap**, mencatat `quant_recipe`).
Output: `<stem>_int8_static.onnx` + `benchmark_int8_static_results.json`.

Model NAS:
```bash
python benchmark_int8_static.py \
    --model_dir nas_results/retrain_repconv_C8_mid14_834cls \
    --onnx_name model_benchmark.onnx \
    --calib_dir preprocessed_results \
    --num_calib 200
```

Baseline MobileNetV3 (nama file ONNX beda → pakai `--onnx_name`):
```bash
python benchmark_int8_static.py \
    --model_dir MobileNetV3Large \
    --onnx_name mobilenetv3_benchmark.onnx \
    --calib_dir preprocessed_results \
    --num_calib 200
```

Cek `quant_recipe` di hasil JSON harus berisi `"per_channel": true` dan `"quant_pre_process": true`.

---

## 5. Akurasi ONNX (FP32 / INT8) — `eval_onnx_accuracy.py`

Top-1 di test split memakai preprocessing proyek (faithful ke `test_results.json`).

```bash
# FP32
python eval_onnx_accuracy.py \
    --onnx nas_results/retrain_repconv_C8_mid14_834cls/model_benchmark.onnx \
    --data_dir /Users/fahmitaris/Downloads/NAS-DARTS-TEMP/preprocessed_results \
    --split_path split_info.json

# INT8
python eval_onnx_accuracy.py \
    --onnx nas_results/retrain_repconv_C8_mid14_834cls/model_benchmark_int8_static.onnx \
    --data_dir /Users/fahmitaris/Downloads/NAS-DARTS-TEMP/preprocessed_results \
    --split_path split_info.json
```
Output: `<onnx_stem>_acc.json`. Gunakan untuk hitung Δakurasi FP32→INT8.

---

## 6. Knowledge Distillation — `knowledge_distilation/kd_train.py`

Teacher (mis. EfficientNet-V2-M) → student (model NAS terpilih).
Jalankan **setelah** arsitektur final ditetapkan (KD = optimasi di atas arsitektur, bukan penentu arsitektur).

```bash
python knowledge_distilation/kd_train.py \
    --student_config nas_results/retrain_repconv_C8_mid14_834cls/config.json \
    --student_weights nas_results/retrain_repconv_C8_mid14_834cls/best_model.pth \
    --teacher_arch efficientnet_v2_m \
    --temperature 8.0 \
    --alpha 0.3 \
    --label_smoothing 0.0 \
    --no_mix \
    --epochs 150 \
    --batch_size 64 \
    --output_dir knowledge_distilation/kd_results/run_repconvC8_mid14_effnet_t8_a0.3_ls0_nomix
```
- `--alpha` = bobot CE; bobot KD = `1 - alpha`.
- `--label_smoothing 0.0` direkomendasikan saat KD aktif.
- `--no_mix` mematikan MixUp/CutMix.

Export ONNX hasil KD — `export_kd_onnx.py`:
```bash
python export_kd_onnx.py \
    --model-dir knowledge_distilation/kd_results/run_repconvC8_mid14_effnet_t8_a0.3_ls0_nomix \
    --opset 13
```
Lalu kuantisasi + eval seperti langkah 4–5.

---

## 7. Benchmark FP32 & INT8 di Pi — `benchmark_compare_onnx_pi.py`

Bandingkan dua ONNX (akurasi + latency) di Raspberry Pi. Jalankan **di Pi**.

INT8 vs INT8 (NAS vs MobileNet):
```bash
python3 benchmark_compare_onnx_pi.py \
    --model-a nas_results/retrain_repconv_C8_mid14_834cls/model_benchmark_int8_static.onnx \
    --label-a "NAS-repconv_C8_mid14-int8" \
    --model-b MobileNetV3Large/mobilenetv3_benchmark_int8_static.onnx \
    --label-b "MobileNetV3Large-int8" \
    --data-dir preprocessed_results \
    --split-path split_info.json \
    --threads 4 \
    --max-samples 834 \
    --save-path benchmark_int8_vs_int8_pi.json
```

FP32 vs FP32: ganti kedua `--model-*` ke `model_benchmark.onnx` / `mobilenetv3_benchmark.onnx`
dan `--save-path benchmark_fp32_vs_fp32_pi.json`.

Flag lain: `--skip-accuracy-a/-b` (latency-only), `--subject-map-a/-b` (mapping subjek khusus),
`--warmup`, `--input-size`.

> 🔴 **GOTCHA file basi:** ukuran ONNX adalah tell-tale. Per-tensor MobileNet = **5.544 MB**,
> per-channel (benar) = **5.798 MB**. Kalau hasil Pi aneh (mis. akurasi 81%), **cek ukuran file**
> — kemungkinan Pi masih pakai file lama. Sinkronkan file yang benar + `benchmark_int8_static.py`
> versi hardening + ONNX opset-13 ke Pi sebelum re-quantize.

---

## Catatan Kuantisasi — Pitfall

1. **Selalu export opset ≥ 13** (semua script export default 13).
2. **Per-channel wajib**; `benchmark_int8_static.py` kini *error keras* bila gagal, tidak diam-diam turun ke per-tensor.
3. **`quant_pre_process`** (shape inference + cleanup) penting untuk graf kompleks (SE/h-swish MobileNetV3).
4. **Resep identik** untuk semua model agar perbandingan adil (QDQ, QInt8/QInt8, per-channel).
5. Untuk paper: sertakan baris **ablasi per-tensor vs per-channel** (file `MobileNetV3Large/mobilenetv3_int8_PERTENSOR_opset12_UNFAIR.onnx`) untuk menunjukkan kamu paham pitfall PTQ.

---

## Hasil terverifikasi (referensi, test set 834 kelas, Pi)

Semua INT8 = per-channel (fair), semua ONNX di-export `export_retrain_run6_plus2_onnx.py` (opset 13).
Konfigurasi spasial sama: `stem_downsample=4`, `reduction_indices=2,5`, genotype `search_mobile_v2`.

| Model | Op | C | Params | FP32 acc | FP32 size | FP32 lat | INT8 acc | INT8 size | INT8 lat |
|---|---|---|---|---|---|---|---|---|---|
| mbconv_C4_stemds4 | mbconv | 4 | 238,720 | 97.24% | 0.562 MB | **4.69 ms** | 97.72% | 0.597 MB | 5.80 ms |
| mbconv_C6_stemds4 | mbconv | 6 | 338,362 | **99.28%** | 0.937 MB | 7.16 ms | 99.28% | 0.720 MB | 8.14 ms |
| mbconv_C8_stemds4 | mbconv | 8 | 461,084 | **99.40%** | 1.404 MB | 10.07 ms | 99.28% | 0.868 MB | 8.36 ms |
| repconv_C8_mid14 | repconv | 8 | 503,228 | 98.80% | 1.457 MB | 6.20 ms | 98.92% | 0.599 MB | **5.47 ms** |
| MobileNetV3Large | — | — | ~5.4M | 99.88% | 21.08 MB | ~15.5 ms | 98.68% | 5.80 MB | 8.46 ms |

**Temuan kunci:**
1. **Spatial reduction = lever latency dominan, hampir accuracy-neutral.** mbconv C4: stem=2 → 20.46 ms/98.08%; stem=4 → 4.69 ms/97.24% (~4.4× lebih cepat).
2. **mbconv > repconv pada FP32** (iso-config C8): 99.40% vs 98.80%, params lebih kecil. Insting awal C4-mbconv terbukti.
3. **INT8 hanya menguntungkan repconv.** repconv: FP32 6.20→INT8 5.47 ms (turun). mbconv C4/C6: INT8 **lebih lambat** dari FP32 (overhead QDQ pada depthwise > hemat komputasi). repconv fuse ke 1 conv padat → INT8 GEMM benar-benar lebih cepat.
4. **Pilihan operator bergantung presisi deploy:** FP32 → mbconv (mbconv_C6: 99.28%, 0.94 MB, 7.16 ms). INT8 → repconv (98.92%, 0.60 MB, 5.47 ms).
5. Semua model NAS jauh lebih kecil (0.56–1.46 MB vs 21 MB) & lebih cepat dari MobileNetV3Large; akurasi mbconv_C8 (99.40%) hanya −0.48% dari MobileNet.

> ⚠️ Single seed (42). Untuk klaim "mbconv > repconv 0.6%" perlu ≥3 seed + uji McNemar (paired) karena selisihnya ~5 sampel.

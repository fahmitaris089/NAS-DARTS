# PalmVein Lightweight Benchmark

Repositori eksperimen mandiri untuk membandingkan arsitektur ringan pada split palm vein yang sama. Kode di folder ini tidak mengimpor modul proyek tesis di direktori induk. Satu-satunya sumber daya eksternal lokal adalah citra pada `../preprocessed_results`, yang memang tidak disalin. Model `mnasnet_b1_torchvision` dipertahankan sebagai eksperimen transfer tambahan agar hasil lama tidak hilang setelah koreksi identitas varian MnasNet.

## Model

- `proxylessnas_mobile`
- `fbnet_c`
- `mnasnet_a1`
- `ampvnet`
- `mnasnet_b1_torchvision` (hasil transfer lama; bukan pembanding scratch utama)
- `ding_baseline`
- `ding_pw`
- `ding_pruned`
- `pdarts_l005_c12_cells10`
- `palmnet_05x_2413`
- `palmnet_05x_2411`

MnasNet-A1 merupakan implementasi lokal berdasarkan struktur A1 yang diterbitkan pada paper dan digunakan hanya pada protokol scratch. MnasNet-B1 mempertahankan topologi `torchvision.mnasnet1_0` serta hasil transfer lama. Tiga model Ding dan dua model PalmNet utama merupakan rekonstruksi independen yang dibatasi oleh struktur paper, bukan implementasi resmi. Implementasi Ding lima blok yang lama diarsipkan dan tidak masuk ringkasan utama. Status, sumber, perubahan classifier, dan asumsi teknis dijelaskan dalam [`docs/MODEL_PROVENANCE.md`](docs/MODEL_PROVENANCE.md).

Varian PalmNet lain yang tercantum pada tabel paper juga dapat dipilih secara eksplisit: `palmnet_05x_2223`, `palmnet_05x_4223`, `palmnet_05x_6223`, `palmnet_05x_2323`, `palmnet_05x_2313`, `palmnet_05x_2412`, `palmnet_10x_2413`, dan `palmnet_20x_2413`. Varian tersebut tidak dijalankan oleh opsi `--models all`.

Untuk menjalankan ulang ketiga rekonstruksi Ding setelah validasi arsitektur:

```bash
python scripts/run_experiments.py \
  --protocol scratch \
  --models ding_baseline ding_pw ding_pruned \
  --seeds 42 123 2026
```

## Persiapan

```bash
cd PalmVein_Lightweight_Benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_dataset.py
python scripts/validate_models.py
```

`mobile-cv` diperlukan hanya untuk bobot pralatih resmi FBNet-C. Seluruh model scratch dapat dibangun tanpa mengimpor paket tersebut.

## Pelatihan

Satu eksperimen scratch:

```bash
python scripts/train.py --model fbnet_c --protocol scratch --seed 42
```

Matriks utama, sembilan model × tiga seed:

```bash
python scripts/run_experiments.py \
  --protocol scratch \
  --models all \
  --seeds 42 123 2026
```

Analisis pralatih terpisah:

```bash
python scripts/run_experiments.py \
  --protocol pretrained \
  --models proxylessnas_mobile fbnet_c mnasnet_b1_torchvision \
  --seeds 42 123 2026
```

Pemanggilan pretrained untuk Ding atau P-DARTS berhenti dengan pesan `N/A`; kode tidak membuat bobot pralatih sintetis.

PalmNet hanya mendukung protokol scratch. Eksperimen dua konfigurasi utamanya dapat dijalankan secara terpisah setelah validasi:

```bash
python scripts/run_experiments.py \
  --protocol scratch \
  --models palmnet_05x_2413 palmnet_05x_2411 \
  --seeds 42 123 2026
```

## ONNX, PTQ, dan Raspberry Pi

```bash
python scripts/export_onnx.py \
  --checkpoint artifacts/checkpoints/scratch/fbnet_c/seed_42/best.pth

python scripts/quantize_int8.py \
  --onnx artifacts/onnx_fp32/scratch_fbnet_c_seed42.onnx

python scripts/benchmark_raspberry_pi.py \
  --onnx artifacts/onnx_int8/scratch_fbnet_c_seed42_int8_qdq.onnx

python scripts/summarize_results.py
```

`benchmark_raspberry_pi.py` mencatat platform aktual dan hanya menandai hasil sebagai layak diklaim berasal dari Raspberry Pi apabila mesin ARM64 Linux terdeteksi. Menjalankannya di laptop berguna untuk smoke test, tetapi bukan bukti latency Raspberry Pi 5.

## Artefak

- konfigurasi dan log per epoch: `results/{scratch,pretrained}/<model>/seed_<seed>/`;
- checkpoint terbaik/terakhir: `artifacts/checkpoints/`;
- ONNX FP32 dan INT8: `artifacts/onnx_fp32/`, `artifacts/onnx_int8/`;
- CSV ringkasan: `results/summary/`;
- validasi spesifikasi: `results/model_validation/model_spec_validation.csv`.

Protokol ilmiah lengkap tersedia di [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

# PalmVein Lightweight Benchmark

Repositori eksperimen mandiri untuk membandingkan tujuh arsitektur ringan pada split palm vein yang sama. Kode di folder ini tidak mengimpor modul proyek tesis di direktori induk. Satu-satunya sumber daya eksternal lokal adalah citra pada `../preprocessed_results`, yang memang tidak disalin.

## Model

- `proxylessnas_mobile`
- `fbnet_c`
- `mnasnet_a1`
- `ding_baseline`
- `ding_pw`
- `ding_pruned`
- `pdarts_l005_c12_cells10`

Tiga model Ding adalah **paper-constrained independent reconstruction**, bukan implementasi resmi. Status, sumber, perubahan classifier, dan asumsi teknis dijelaskan dalam [`docs/MODEL_PROVENANCE.md`](docs/MODEL_PROVENANCE.md).

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

Matriks utama, tujuh model × tiga seed:

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
  --models proxylessnas_mobile fbnet_c mnasnet_a1 \
  --seeds 42 123 2026
```

Pemanggilan pretrained untuk Ding atau P-DARTS berhenti dengan pesan `N/A`; kode tidak membuat bobot pralatih sintetis.

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

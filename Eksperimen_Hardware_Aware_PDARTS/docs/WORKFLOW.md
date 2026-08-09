# Workflow Eksperimen

## 1. Preprocessing

`scripts/01_preprocess.sh` mengubah citra mentah SCUT_PV_v1 menjadi input
224 x 224 yang digunakan oleh seluruh eksperimen.

## 2. Latency LUT

Jalankan ekspor probe pada komputer training, salin probe ke Raspberry Pi,
lalu ukur dengan `scripts/02_build_lut.sh measure`. LUT INT8 final tersedia
di `results/lut/`.

## 3. Hardware-Aware P-DARTS

`scripts/03_search_pdarts.sh <lambda>` menjalankan satu pencarian. Nilai lambda
tesis adalah 0.00, 0.05, 0.10, dan 0.20. Search menghasilkan genotype, bukan
checkpoint model final.

## 4. Retraining dan Refinement

`scripts/04_retrain.sh` membangun genotype diskrit L0.05 dengan C_init=12 dan
10 cell. Bukti konfigurasi lain yang dilaporkan tersedia di
`results/retraining/`.

## 5. Teacher

`scripts/05_train_teacher.sh` melatih EfficientNetV2M. Ringkasan seluruh
14 eksperimen teacher tersedia di `results/teacher/`, termasuk konfigurasi,
log pelatihan, metrik pengujian, laporan klasifikasi, dan visual evaluasi.
Delapan kandidat yang masuk Tabel 4.11 dipetakan secara khusus pada
`results/thesis_manifest.csv`. Checkpoint yang disertakan hanya EfficientNetV2M
karena model tersebut dipilih sebagai teacher pada tahap KD.

## 6. Knowledge Distillation

`scripts/06_run_kd.sh` menjalankan konfigurasi final T=20 dan alpha=0.5.
Bukti seluruh skenario KD tesis tersedia di `results/kd/`.

## 7-9. Deployment

Ekspor, PTQ, dan benchmark dijalankan berurutan dengan skrip 07-09. Model
final siap uji juga sudah tersedia pada `models/`.

# Repositori Final Eksperimen Tesis

Repositori ini memuat kode, konfigurasi, bukti hasil, checkpoint penting, dan
model deployment untuk tesis palm-vein recognition berbasis hardware-aware
P-DARTS, Knowledge Distillation (KD), dan Post-Training Quantization (PTQ).
Seleksi eksperimen dan angka final mengacu pada dokumen tesis terbaru:
`backup laporan tesisi/Draft BACKUP_Mohammad_Taris_Syahir_Zul_Fahmi_6025242008_Final.docx`
di repositori sumber.

## Alur Utama

1. Preprocessing SCUT_PV_v1
2. Penyusunan latency LUT pada Raspberry Pi 5
3. Pencarian hardware-aware P-DARTS
4. Retraining dan refinement genotype
5. Pelatihan teacher
6. Knowledge Distillation
7. Ekspor ONNX FP32
8. PTQ INT8
9. Benchmark Raspberry Pi 5

Dataset tidak disertakan. Letakkan dataset mentah pada `dataset/raw/` mengikuti
petunjuk di `docs/DATASET_SETUP.md`. Hasil eksperimen yang dilaporkan dalam
tesis tersedia di `results/`; checkpoint penting tersedia di `checkpoints/`.
Folder `results/teacher/` juga memuat artefak ringkas seluruh 14 eksperimen
teacher yang pernah dijalankan. Delapan kandidat yang dipertahankan pada tabel
final tesis dibedakan melalui `results/thesis_manifest.csv`.

## Mulai Cepat

```bash
cd palm-vein-hardware-aware-pdarts
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/training.txt
python tests/smoke_test.py
```

Perintah setiap tahap berada di `scripts/`. Pencarian, retraining, teacher,
dan KD memerlukan GPU untuk waktu eksekusi yang wajar. Penyusunan ulang LUT
dan benchmark final harus dijalankan pada Raspberry Pi 5 agar sesuai dengan
lingkungan tesis.

Dokumentasi rinci:

- `docs/WORKFLOW.md`: urutan dan keluaran setiap tahap.
- `docs/DATASET_SETUP.md`: struktur dataset dan split.
- `docs/HARDWARE.md`: lingkungan GPU dan Raspberry Pi.
- `docs/RESULTS_MAPPING.md`: hubungan hasil repositori dengan tabel tesis.
- `docs/REPRODUCIBILITY.md`: batas reproduksi dan pemeriksaan integritas.

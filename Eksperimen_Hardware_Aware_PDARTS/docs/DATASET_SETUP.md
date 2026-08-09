# Penyiapan Dataset

Dataset SCUT_PV_v1 tidak didistribusikan dalam repositori ini.

Susunan masukan yang diharapkan:

```text
dataset/raw/
├── 1/
│   ├── 1_1.bmp
│   └── ...
├── 2/
└── ...
```

Jalankan:

```bash
scripts/01_preprocess.sh
```

Hasil disimpan pada `dataset/preprocessed/<subject_id>/`. Pipeline melakukan
deteksi pusat telapak berbasis gradien, refinement centroid berbobot
intensitas, crop ROI 384 x 384, CLAHE, normalisasi, dan resize 224 x 224.

`dataset/splits/split_info.json` adalah split yang digunakan pada eksperimen
tesis. Jangan membuat split baru ketika mereproduksi angka tesis.

`dataset/calibration/` digunakan oleh PTQ. Isinya harus berupa citra hasil
preprocessing yang representatif. Repositori ini menggunakan 834 citra dari
training split, yaitu satu citra untuk setiap kelas. Daftar sumber dan lokasi
setiap citra disimpan pada `dataset/calibration/calibration_manifest.csv`.
Data kalibrasi hanya digunakan untuk mengamati rentang aktivasi dan tidak
melibatkan label, perhitungan loss, atau pembaruan bobot.

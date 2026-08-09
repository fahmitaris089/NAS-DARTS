# Dataset linkage

Dataset citra tidak disalin. `configs/dataset.json` menunjuk ke `../preprocessed_results` relatif terhadap root benchmark, yaitu folder `preprocessed_results` pada repositori NAS-DARTS.

Split adalah salinan byte-identik dari `../split_info.json` pada saat benchmark dibuat:

- SHA-256: `8e393a52fbc93c19d420c942adf104b1910c708e796fcdb13e17ac90482966de`;
- training: 6.672 citra;
- validation: 834 citra;
- test: 834 citra;
- kelas: 834;
- overlap antar-split: 0.

Jalankan `python scripts/prepare_dataset.py` sebelum eksperimen. Perintah tersebut memvalidasi seluruh path citra dan membuat ulang `calibration_manifest.json` dari training split saja. Manifest memilih satu nama berkas pertama secara leksikografis untuk setiap subjek numerik dan menyimpan hash citra.

Kode ROI, CLAHE, normalisasi intensitas, dan resize tersedia di `src/data/preprocessing.py` untuk dokumentasi reproduksibilitas. Training tidak menjalankan preprocessing ini karena input benchmark adalah dataset yang sudah diproses. Menjalankannya kembali akan menghasilkan kondisi data berbeda dan harus diperlakukan sebagai eksperimen baru.

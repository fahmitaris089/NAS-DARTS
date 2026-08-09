# Reproduksibilitas

Repositori memisahkan:

- `src/` dan `scripts/`: kode yang dapat dijalankan ulang.
- `configs/`: konfigurasi portabel yang merangkum eksperimen tesis.
- `results/`: bukti metrik dan log dari eksperimen yang telah dijalankan.
- `checkpoints/` dan `models/`: artefak final yang diperlukan untuk evaluasi.

Beberapa log arsip dapat menyebut struktur folder mesin eksperimen lama.
Rujukan tersebut hanya merupakan provenance log dan tidak dipakai oleh skrip
portabel. Semua entry point pada `scripts/` membentuk jalur dari akar
repositori.

Gunakan `python tests/smoke_test.py` untuk memeriksa struktur, JSON, genotype,
checkpoint, dan model ONNX. Gunakan `shasum -a 256 -c MANIFEST.sha256` untuk
memeriksa integritas setelah repositori dipindahkan.

Pelatihan penuh tidak dapat direproduksi tanpa dataset dan perangkat training.
Pengukuran LUT serta latency final hanya setara dengan tesis jika dijalankan
pada lingkungan Raspberry Pi yang sama.


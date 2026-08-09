# Dataset setup and leakage controls

Konfigurasi dataset terdapat pada `configs/dataset.json`. Path relatif selalu diselesaikan terhadap root benchmark, bukan current working directory. Dengan demikian, perintah boleh dijalankan dari folder mana pun.

`scripts/prepare_dataset.py` memeriksa:

1. jumlah split 6.672/834/834;
2. total 8.340 path unik dan 834 subjek;
3. tidak ada pasangan `(subject, filename)` yang muncul di dua split;
4. prefix nama berkas sesuai ID subjek;
5. seluruh file terdapat di disk;
6. hash split;
7. satu entry kalibrasi per kelas dan seluruhnya merupakan anggota training split.

Split validation dan test masing-masing memuat satu citra per kelas. Karena itu, akurasi closed-set yang sangat tinggi tetap memiliki ketidakpastian dan sensitivitas terhadap satu citra per identitas. Benchmark tidak menyediakan EER/TAR karena split dan classifier ini dirancang untuk closed-set identification, bukan protokol verification/open-set.

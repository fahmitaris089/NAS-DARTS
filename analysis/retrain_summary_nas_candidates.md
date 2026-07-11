# Summary Hasil Retraining Kandidat NAS

Dokumen ini merangkum hasil retraining kandidat NAS untuk kebutuhan penyusunan Bab 4. Fokus bagian ini adalah membaca kecenderungan performa arsitektur setelah genotype hasil NAS diretraining sebagai model utuh. Oleh karena itu, bagian ini belum diposisikan sebagai pemilihan model akhir, tetapi sebagai dasar evaluasi untuk tahap refinement, Knowledge Distillation, dan evaluasi INT8.

## 1. Setup Umum Retraining

Mayoritas kandidat awal diretraining dengan konfigurasi yang sama agar perbandingan lebih adil:

| Komponen | Konfigurasi |
|---|---|
| Dataset | Palm-vein, 834 kelas |
| Split | Train 6672, val 834, test 834 |
| Resolusi input | 224 x 224 |
| Stem downsample | 8 |
| Num cells | 8 |
| Reduction indices | 2,5 |
| Epoch | 300 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Weight decay | 0.05 |
| Drop path | 0.2 |
| Augmentasi | v1_legacy + CutOut 16 |
| Auxiliary head | Aktif saat retraining |
| Seed | 42 |

Catatan: nilai parameter pada tabel berikut dihitung pada model inference tanpa auxiliary head. FLOPs dan akurasi retraining berasal dari evaluasi checkpoint retraining. Ukuran model pada kolom `Size` mengacu pada ukuran checkpoint PyTorch yang tersimpan pada `test_results.json`, sedangkan ukuran ONNX/INT8 untuk deployment dapat berbeda karena format penyimpanan dan graph export.

## 2. Hasil Retraining Kandidat NAS Awal

Tabel ini menggabungkan hasil kandidat awal untuk `lambda = 0.00`, `0.05`, `0.10`, dan `0.20` pada variasi `C_init = 6, 8, 10`. Untuk `lambda = 0.05` dan `0.20`, angka benchmark Raspberry Pi mengikuti hasil yang sudah digunakan pada tabel eksperimen sebelumnya. Untuk `lambda = 0.00` dan `0.10`, angka Raspberry Pi mengikuti hasil uji terbaru.

| Lambda | C_init | Test Acc | Params Inference | FLOPs (M) | Size (MB) | Pi FP32 (ms) | Pi INT8 (ms) | INT8 Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 6 | 97.12% | 183,744 | 19.49 | 1.178 | 2.57 | 3.68 | 96.76% |
| 0.00 | 8 | 98.08% | 283,386 | 29.96 | 1.579 | 3.46 | 3.45 | 98.08% |
| 0.00 | 10 | 98.08% | 402,364 | 42.42 | 2.054 | 4.08 | 4.56 | 98.20% |
| 0.05 | 6 | 96.52% | 195,624 | 19.97 | 1.220 | 2.05 | 2.36 | 96.28% |
| 0.05 | 8 | 98.08% | 308,538 | 31.58 | 1.672 | 2.87 | 2.43 | 98.20% |
| 0.05 | 10 | 99.04% | 445,444 | 45.67 | 2.214 | 3.52 | 3.17 | 99.28% |
| 0.10 | 6 | 96.16% | 257,976 | 26.28 | 1.463 | 2.25 | 2.24 | 96.16% |
| 0.10 | 8 | 98.20% | 420,058 | 42.83 | 2.103 | 3.12 | 2.41 | 97.96% |
| 0.10 | 10 | 98.68% | 620,324 | 63.28 | 2.888 | 4.03 | 3.15 | 98.68% |
| 0.20 | 6 | 97.72% | 245,592 | 25.29 | 1.415 | 2.17 | 2.09 | 97.36% |
| 0.20 | 8 | 98.44% | 398,106 | 41.09 | 2.018 | 3.00 | 2.22 | 97.96% |
| 0.20 | 10 | 98.80% | 586,084 | 60.56 | 2.757 | 3.77 | 2.94 | 98.92% |

## 3. Ringkasan Pola dari Kandidat Awal

Beberapa pola penting dapat dibaca dari tabel di atas.

Pertama, peningkatan `C_init` secara umum menaikkan kapasitas model, tetapi kenaikan akurasi tidak selalu linear. Pada `lambda = 0.05`, peningkatan dari C6 ke C10 cukup konsisten, yaitu dari 96.52% menjadi 99.04%. Namun pada `lambda = 0.00`, peningkatan dari C8 ke C10 tidak menaikkan akurasi FP32, karena keduanya tetap berada pada 98.08%.

Kedua, `lambda = 0.00` tidak otomatis menghasilkan model dengan akurasi tertinggi. Meskipun konfigurasi ini tidak memberi penalti eksplisit terhadap *latency*, hasil retraining menunjukkan bahwa performa akhirnya tetap dipengaruhi oleh genotype yang terbentuk, kapasitas model, dan proses optimisasi saat retraining. Temuan ini penting agar pembahasan tidak menyederhanakan `lambda = 0.00` sebagai konfigurasi yang pasti paling akurat tetapi paling lambat.

Ketiga, efek INT8 berbeda antar genotype. Pada `lambda = 0.00`, INT8 tidak selalu mempercepat inferensi di Raspberry Pi. C6 dan C10 bahkan menunjukkan INT8 lebih lambat dibanding FP32. Sebaliknya, pada `lambda = 0.05`, `0.10`, dan `0.20`, konfigurasi C8 atau C10 cenderung memperoleh percepatan INT8 yang lebih jelas. Ini mengindikasikan bahwa efisiensi INT8 bukan hanya ditentukan oleh jumlah parameter, tetapi juga oleh komposisi operator dan struktur graph ONNX yang dihasilkan.

Keempat, kandidat `L0.05 C10` masih menjadi salah satu titik kuat pada tahap awal karena mencapai akurasi FP32 99.04% dan INT8 99.28%, dengan *latency* INT8 3.17 ms. Namun, `L0.10 C10` menunjukkan bahwa genotype lain juga kompetitif dari sisi *latency* FP32/INT8, meskipun akurasinya lebih rendah pada hasil terbaru ini. Karena itu, pemilihan kandidat tidak sebaiknya dilakukan hanya dari satu kolom metrik.

## 4. Hasil Refinement Kapasitas Kandidat NAS

Setelah evaluasi kandidat awal, dilakukan refinement kapasitas pada genotype `lambda = 0.05`. Refinement ini menguji apakah peningkatan kedalaman (`num_cells`) atau lebar kanal (`C_init`) dapat memberikan akurasi lebih tinggi sebelum model digunakan pada tahap Knowledge Distillation.

| Model | C_init | Cells | Test Acc | Params Inference | FLOPs (M) | Size (MB) | Catatan |
|---|---:|---:|---:|---:|---:|---:|---|
| L0.05 C12 cells10 | 12 | 10 | 98.92% | 637,842 | 72.67 | 2.926 | Kandidat seimbang untuk tahap KD |
| L0.05 C12 cells11 | 12 | 11 | 98.92% | 734,946 | 77.49 | 3.348 | Lebih dalam, belum meningkatkan akurasi |
| L0.05 C14 cells10 | 14 | 10 | 99.40% | 833,722 | 95.25 | 3.686 | Akurasi retraining tertinggi, tetapi lebih berat |

Benchmark tambahan untuk `L0.05 C14 cells10` pada Raspberry Pi menunjukkan FP32 99.40% dengan ukuran ONNX 3.167 MB dan *latency* 6.25 ms. Setelah INT8, akurasi menjadi 99.28%, ukuran 1.123 MB, dan *latency* 4.49 ms. Hasil ini memperjelas bahwa peningkatan `C_init` ke 14 memang menaikkan akurasi retraining, tetapi biaya deployment-nya menjadi lebih dekat dengan MobileNetV3Small. Karena tujuan penelitian tidak hanya mengejar akurasi retraining, kandidat ini perlu dibaca sebagai pembanding kapasitas, bukan otomatis sebagai model akhir.

## 5. Implikasi untuk Narasi Bab 4

Untuk Bab 4, pembahasan yang paling aman secara metodologis adalah memisahkan tiga tahap:

1. Tahap genotype dan *latency-aware search* menjelaskan bagaimana perubahan `lambda` memengaruhi kecenderungan topologi.
2. Tahap retraining kandidat awal melaporkan performa aktual genotype setelah dilatih sebagai model lengkap pada variasi `C_init`.
3. Tahap refinement kapasitas menguji kandidat yang dianggap menjanjikan pada kapasitas lebih besar sebelum masuk ke Knowledge Distillation.

Dengan struktur ini, narasi tidak perlu memaksakan bahwa satu lambda sudah menjadi pemenang sejak awal. Data justru menunjukkan bahwa performa akhir dipengaruhi oleh kombinasi `lambda`, `C_init`, genotype, stabilitas INT8, dan biaya inferensi pada Raspberry Pi.

## 6. Draf Narasi Singkat untuk Bab 4

Berikut contoh narasi yang dapat digunakan atau disesuaikan:

> Tabel 4.x memperlihatkan bahwa peningkatan kapasitas melalui `C_init` cenderung meningkatkan akurasi, tetapi pola peningkatannya tidak sepenuhnya linear pada semua nilai `lambda`. Pada `lambda = 0.05`, konfigurasi C10 memperoleh akurasi tertinggi pada kelompok kandidat awal, yaitu 99.04%, serta tetap stabil setelah kuantisasi INT8 dengan akurasi 99.28%. Namun, hasil pada `lambda = 0.00` dan `lambda = 0.10` menunjukkan bahwa performa akhir tidak dapat dijelaskan hanya dari besar kecilnya penalti *latency*. Genotype yang terbentuk, jumlah parameter, FLOPs, serta kompatibilitas operator terhadap eksekusi INT8 pada Raspberry Pi turut memengaruhi hasil akhir.

> Hasil benchmark Raspberry Pi juga menunjukkan bahwa INT8 tidak selalu memberikan percepatan pada semua kandidat. Pada beberapa konfigurasi `lambda = 0.00`, model INT8 justru memiliki *latency* lebih tinggi dibanding FP32. Kondisi ini mengindikasikan bahwa efisiensi model pada perangkat *edge* tidak cukup dinilai dari ukuran model atau jumlah FLOPs saja, tetapi perlu diverifikasi melalui pengujian langsung pada perangkat target.

> Berdasarkan evaluasi awal tersebut, refinement kapasitas dilakukan untuk menguji apakah peningkatan `num_cells` dan `C_init` dapat menghasilkan kandidat student yang lebih kuat sebelum tahap Knowledge Distillation. Hasil refinement menunjukkan bahwa C14 cells10 mencapai akurasi retraining tertinggi, tetapi memiliki jumlah parameter, FLOPs, dan *latency* Raspberry Pi yang lebih besar. Sebaliknya, C12 cells10 menyediakan kompromi kapasitas yang lebih seimbang untuk tahap lanjutan, terutama karena tujuan penelitian tidak hanya mengejar akurasi FP32, tetapi juga mempertahankan efisiensi dan stabilitas setelah quantization.

## 7. Catatan untuk Tabel Tesis

- Jika tabel utama terlalu panjang, tampilkan `lambda = 0.00`, `0.05`, `0.10`, dan `0.20` dengan C6/C8/C10 sebagai tabel utama retraining.
- Tabel refinement sebaiknya dipisah dari tabel kandidat awal agar alasan munculnya C12 cells10 tidak terlihat tiba-tiba.
- Jangan menyebut `L0.05 C12 cells10` sebagai kandidat terbaik pada subbab retraining awal. Lebih aman menyebutnya sebagai kandidat refinement yang dipilih untuk tahap KD karena kompromi kapasitas, kompleksitas, dan hasil deployment.
- Untuk klaim final, gunakan hasil setelah KD dan QAT, bukan hanya retraining.

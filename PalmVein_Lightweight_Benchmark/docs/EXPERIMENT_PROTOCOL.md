# Controlled experiment protocol

## Pertanyaan eksperimen

Benchmark utama menguji perbedaan kualitas arsitektur ketika sumber bobot awal, split, preprocessing input, augmentasi, optimizer, jadwal learning rate, jumlah epoch, dan aturan pemilihan checkpoint dikendalikan. Hasil tidak membuktikan keunggulan universal di luar dataset dan perangkat yang diuji.

## Scratch primer

- 600 epoch;
- seed 42, 123, dan 2026;
- AdamW, learning rate awal `1e-3`, weight decay `0.05`;
- warm-up 10 epoch dari faktor `0.01`, lalu cosine annealing sampai `1e-6`;
- batch size 64;
- label smoothing 0,2;
- gradient clipping norm 1,0;
- input 224 × 224, grayscale direplikasi menjadi RGB, normalisasi ImageNet;
- augmentasi ringan identik: rotasi 5°, translasi 3%, skala 0,97–1,08, brightness 0,08, contrast 0,05;
- tidak ada horizontal flip karena tangan kiri dan kanan merupakan identitas berbeda;
- checkpoint dipilih hanya berdasarkan validation loss minimum;
- test set dievaluasi satu kali setelah checkpoint terpilih.

P-DARTS menggunakan `drop_path_probability=0` dan tanpa auxiliary loss agar protokol tidak memberi regularizer tambahan yang tidak dinyatakan kepada satu model saja. Keputusan ini menjaga kontrol eksperimen, tetapi berbeda dari resep retraining DARTS yang lazim dan harus disebutkan saat membandingkan dengan angka tesis lama.

Matriks scratch utama memuat P-DARTS L0.05, ProxylessNAS-Mobile, FBNet-C, MnasNet-A1, tiga tahap rekonstruksi Ding, serta `PalmNet-0.5x2413` dan `PalmNet-0.5x2411`. Dua PalmNet tersebut adalah rekonstruksi independen berbasis paper. Varian PalmNet lain dapat dibangun melalui CLI, tetapi tidak dijalankan oleh pilihan `--models all`. Chen StudentNet tetap menjadi artefak audit yang tidak terdaftar dan tidak masuk tabel benchmark utama.

## Pretrained sekunder

Hanya ProxylessNAS-Mobile, FBNet-C, dan MnasNet-B1 (`torchvision.mnasnet1_0`). Backbone dibekukan lima epoch. Classifier memakai learning rate `1e-3`; setelah unfreeze, backbone memakai `1e-4`. Pelatihan berlangsung 200 epoch dengan aturan checkpoint/test yang sama. Hasil ini dilaporkan terpisah karena mengukur transfer learning, bukan kualitas arsitektur dari inisialisasi yang setara. MnasNet-A1 dan seluruh PalmNet tidak dimasukkan karena tidak tersedia bobot PyTorch resmi yang telah diaudit untuk arsitektur yang tepat.

## Agregasi statistik

CSV scratch dan pretrained melaporkan mean serta sample standard deviation (`n-1`) dari akurasi tiga seed. Jika baru satu atau dua seed selesai, `seeds_completed` memperlihatkan ketidaklengkapan; tabel tidak boleh dinarasikan seolah tiga seed telah tersedia.

## Deployment

- ONNX opset 13 dan dynamic batch axis;
- validasi `onnx.checker` serta perbandingan keluaran PyTorch–ONNX;
- PTQ statis QDQ, QInt8 weight per-channel, QUInt8 activation, MinMax calibration;
- satu manifest kalibrasi bersama: 834 citra, satu per kelas, training split saja;
- ONNX Runtime CPU, intra-op 4 thread, inter-op 1, sequential execution;
- benchmark default 50 warm-up dan 500 iterasi, batch size 1;
- latency dilaporkan sebagai mean, median, dan p95.

FLOPs/MMACs adalah proxy kompleksitas, bukan pengganti latency. Klaim deployment Raspberry Pi hanya sah untuk JSON yang dibuat pada perangkat ARM64 Linux tersebut dengan kondisi daya/termal yang dicatat secara terpisah.

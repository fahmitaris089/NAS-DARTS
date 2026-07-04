# Draft Revisi Bab 1

## Judul yang Direkomendasikan

**Arsitektur Jaringan Ringan untuk Pengenalan Palm Vein pada Perangkat Edge Menggunakan Hardware-Aware Neural Architecture Search dan Knowledge Distillation**

Judul ini mempertahankan fokus lama pada `Neural Architecture Search` dan `Knowledge Distillation`, tetapi menambahkan arah penelitian terbaru, yaitu perancangan arsitektur yang sadar terhadap karakteristik perangkat edge. Istilah `hardware-aware` perlu muncul karena kontribusi utama penelitian tidak hanya terletak pada pencarian arsitektur, tetapi juga pada penggunaan latency lookup table yang diukur pada Raspberry Pi sebagai bagian dari proses search.

---

# BAB 1  
# PENDAHULUAN

## 1.1 Latar Belakang

Sistem biometrik berbasis palm vein memanfaatkan pola pembuluh darah pada telapak tangan sebagai karakteristik identitas. Pola vena berada di bawah permukaan kulit dan umumnya ditangkap menggunakan citra near-infrared (NIR), sehingga relatif sulit ditiru secara visual dibandingkan karakteristik biometrik yang berada di permukaan, seperti wajah atau sidik jari. Karakteristik ini membuat palm vein recognition menarik untuk dikaji pada sistem identifikasi yang membutuhkan tingkat keamanan tinggi dan kenyamanan penggunaan.

Perkembangan deep learning, khususnya Convolutional Neural Network (CNN), telah meningkatkan kemampuan sistem pengenalan biometrik dalam mengekstraksi fitur visual secara otomatis. Pada palm vein recognition, CNN dapat mempelajari representasi pola vena dari citra yang telah melalui proses preprocessing, tanpa harus bergantung sepenuhnya pada fitur buatan tangan. Namun, peningkatan akurasi pada model CNN sering kali diikuti oleh bertambahnya jumlah parameter, kebutuhan komputasi, dan ukuran model. Kondisi tersebut menjadi tantangan ketika sistem ingin diterapkan pada perangkat edge yang memiliki sumber daya komputasi lebih terbatas dibandingkan komputer desktop atau server GPU.

Perangkat edge seperti Raspberry Pi memiliki potensi untuk digunakan pada sistem biometrik portabel karena ukurannya kecil, konsumsi dayanya relatif rendah, dan dapat menjalankan inferensi secara lokal. Inferensi lokal juga mengurangi ketergantungan pada koneksi jaringan dan dapat meningkatkan privasi karena data biometrik tidak harus dikirim ke server eksternal. Meskipun demikian, deployment model deep learning pada perangkat seperti Raspberry Pi tidak hanya ditentukan oleh akurasi model. Latency inferensi, ukuran model, konsumsi memori, dan dukungan runtime menjadi faktor yang sama pentingnya, terutama jika sistem ditujukan untuk penggunaan praktis.

Pada banyak penelitian, efisiensi model sering diukur menggunakan jumlah parameter atau FLOPs. Kedua metrik tersebut penting, tetapi belum selalu menggambarkan latency aktual pada perangkat target. Dua model dengan FLOPs yang mirip dapat memiliki waktu inferensi berbeda karena perbedaan jenis operator, pola akses memori, kemampuan optimasi kernel, dan overhead runtime. Dengan demikian, desain arsitektur yang hanya mengejar model kecil secara teoretis belum tentu menghasilkan model yang paling cepat pada perangkat edge. Hal ini menjadi dasar perlunya pendekatan yang mempertimbangkan karakteristik hardware secara langsung.

Neural Architecture Search (NAS) menawarkan pendekatan otomatis untuk mencari arsitektur jaringan yang sesuai dengan suatu tugas. Dibandingkan desain arsitektur secara manual, NAS dapat mengeksplorasi kombinasi operator dan koneksi dalam search space tertentu. Pada penelitian ini, NAS diarahkan tidak hanya untuk memperoleh arsitektur yang akurat, tetapi juga untuk mempertimbangkan efisiensi inferensi pada perangkat target. Metode yang digunakan adalah Progressive Differentiable Architecture Search (P-DARTS), yaitu pengembangan dari DARTS yang melakukan pencarian arsitektur secara progresif untuk mengurangi perbedaan kompleksitas antara tahap search dan tahap evaluasi akhir.

Komponen penting dalam penelitian ini adalah integrasi latency lookup table (LUT) ke dalam objective NAS. LUT berisi estimasi biaya latency operator yang diukur pada Raspberry Pi. Dengan informasi tersebut, proses search dapat memberikan penalti terhadap operator yang cenderung mahal pada perangkat target. Pendekatan ini menjadikan proses pencarian arsitektur bersifat hardware-aware, karena keputusan arsitektur tidak hanya dipandu oleh classification loss, tetapi juga oleh biaya latency yang diperoleh dari pengukuran perangkat nyata. Dalam eksperimen ini, Raspberry Pi 5 digunakan sebagai target evaluasi deployment dengan ONNX Runtime CPU dan pengukuran latency menggunakan 4 thread.

Setelah arsitektur student diperoleh melalui hardware-aware P-DARTS, penelitian ini juga menerapkan Knowledge Distillation (KD) untuk meningkatkan performa model ringan. KD memanfaatkan informasi dari teacher model berkapasitas lebih besar agar student model dapat mempelajari distribusi prediksi yang lebih informatif daripada label keras saja. Pendekatan ini relevan karena model hasil NAS dirancang agar ringan, sehingga masih terdapat kemungkinan peningkatan akurasi melalui transfer pengetahuan dari teacher.

Tahap akhir penelitian berfokus pada kesiapan deployment melalui export model ke format ONNX dan Post-Training Quantization (PTQ) INT8. Quantization digunakan untuk mengurangi ukuran model dan mempercepat inferensi tanpa melakukan pelatihan ulang penuh. Dengan demikian, penelitian ini tidak hanya mengevaluasi model pada sisi akurasi, tetapi juga pada aspek deployment yang lebih praktis, yaitu ukuran model dan latency inferensi pada Raspberry Pi. Pruning tidak digunakan sebagai metode utama karena model hasil NAS sudah berada pada skala parameter yang kecil, sedangkan PTQ INT8 memberikan jalur kompresi yang lebih langsung untuk evaluasi deployment.

Berdasarkan uraian tersebut, penelitian ini berfokus pada perancangan model palm vein recognition yang akurat dan efisien untuk perangkat edge melalui kombinasi hardware-aware P-DARTS, Knowledge Distillation, dan PTQ INT8. Fokus utama penelitian bukan hanya memperoleh akurasi tertinggi, tetapi mencari trade-off yang layak antara akurasi, ukuran model, dan latency inferensi pada Raspberry Pi.

## 1.2 Rumusan Masalah

Berdasarkan latar belakang tersebut, rumusan masalah dalam penelitian ini adalah sebagai berikut.

1. Bagaimana merancang arsitektur CNN ringan untuk palm vein recognition menggunakan Progressive Differentiable Architecture Search (P-DARTS)?
2. Bagaimana mengintegrasikan latency lookup table Raspberry Pi ke dalam objective NAS agar proses pencarian arsitektur menjadi hardware-aware?
3. Bagaimana performa kandidat arsitektur hasil NAS setelah retraining dari sisi akurasi, jumlah parameter, ukuran model, FLOPs, dan latency inferensi?
4. Bagaimana pengaruh Knowledge Distillation terhadap performa student model hasil hardware-aware NAS?
5. Bagaimana dampak Post-Training Quantization INT8 terhadap akurasi, ukuran model, dan latency inferensi model ketika dievaluasi pada Raspberry Pi?

## 1.3 Tujuan Penelitian

Tujuan penelitian ini adalah sebagai berikut.

1. Mengembangkan arsitektur CNN ringan untuk palm vein recognition menggunakan P-DARTS.
2. Menerapkan hardware-aware NAS dengan memanfaatkan latency lookup table yang diukur pada Raspberry Pi sebagai penalti dalam objective pencarian arsitektur.
3. Mengevaluasi kandidat arsitektur hasil NAS melalui proses retraining dan membandingkan trade-off antara akurasi, kompleksitas model, ukuran model, dan latency.
4. Meningkatkan performa student model hasil NAS menggunakan Knowledge Distillation dari teacher model berkapasitas lebih besar.
5. Mengevaluasi model FP32 dan INT8 setelah export ONNX dan PTQ untuk menilai kelayakan deployment pada Raspberry Pi.

## 1.4 Batasan Masalah

Agar penelitian lebih terarah, batasan masalah dalam penelitian ini ditetapkan sebagai berikut.

1. Penelitian difokuskan pada palm vein recognition berbasis citra NIR yang telah melalui tahap preprocessing ROI, peningkatan kontras, normalisasi, dan resize input.
2. Evaluasi pengenalan dilakukan pada skenario closed-set identification, yaitu kelas pada data uji merupakan kelas yang sama dengan kelas yang tersedia pada tahap pelatihan, tetapi citra uji tidak digunakan dalam proses pelatihan model.
3. Metode NAS yang digunakan adalah P-DARTS dengan search space operator CNN ringan yang telah ditentukan.
4. Komponen hardware-aware pada NAS menggunakan latency lookup table operator yang diukur pada Raspberry Pi. Pengukuran latency end-to-end dilakukan setelah model final diekspor ke ONNX.
5. Raspberry Pi 5 digunakan sebagai target perangkat edge untuk evaluasi deployment. Evaluasi inferensi dilakukan menggunakan ONNX Runtime CPU dengan 4 thread.
6. Model compression pada penelitian ini dibatasi pada Post-Training Quantization INT8. Pruning tidak digunakan sebagai metode utama dalam eksperimen final.
7. Penelitian ini tidak membahas akuisisi citra palm vein secara real-time menggunakan sensor NIR pada Raspberry Pi. Raspberry Pi digunakan sebagai perangkat benchmark inferensi, bukan sebagai perangkat pengambilan citra.

## 1.5 Manfaat Penelitian

Penelitian ini diharapkan memberikan manfaat sebagai berikut.

1. Memberikan rancangan model palm vein recognition yang lebih sesuai untuk perangkat edge karena mempertimbangkan akurasi dan efisiensi inferensi secara bersamaan.
2. Menunjukkan penggunaan hardware-aware NAS berbasis latency lookup table sebagai pendekatan untuk mencari arsitektur yang lebih relevan terhadap performa perangkat nyata.
3. Memberikan gambaran pengaruh Knowledge Distillation terhadap model student hasil NAS pada kasus klasifikasi palm vein.
4. Memberikan evaluasi deployment yang lebih praktis melalui perbandingan model FP32 dan INT8 pada Raspberry Pi.
5. Menjadi referensi eksperimen untuk penelitian lanjutan mengenai biometrik palm vein, NAS, model compression, dan edge AI.

## 1.6 Kontribusi Penelitian

Kontribusi utama penelitian ini adalah sebagai berikut.

1. Mengusulkan pipeline perancangan model palm vein recognition berbasis hardware-aware P-DARTS, sehingga proses pencarian arsitektur mempertimbangkan latency operator pada perangkat target.
2. Menggunakan latency lookup table Raspberry Pi sebagai penalti dalam objective NAS untuk mengarahkan pencarian arsitektur ke model yang lebih sesuai dengan deployment edge.
3. Mengevaluasi kandidat arsitektur hasil NAS pada beberapa konfigurasi kapasitas model untuk memperoleh trade-off antara akurasi dan efisiensi.
4. Menerapkan Knowledge Distillation untuk meningkatkan performa student model hasil NAS tanpa mengubah karakteristik arsitektur utama.
5. Melakukan evaluasi FP32 dan INT8 pada Raspberry Pi menggunakan ONNX Runtime, sehingga klaim efisiensi tidak hanya didasarkan pada parameter atau FLOPs, tetapi juga pada latency inferensi aktual.

## 1.7 Sistematika Penulisan

Sistematika penulisan tesis ini adalah sebagai berikut.

Bab 1 membahas pendahuluan penelitian, meliputi latar belakang, rumusan masalah, tujuan penelitian, batasan masalah, manfaat penelitian, kontribusi penelitian, dan sistematika penulisan.

Bab 2 membahas kajian pustaka dan dasar teori yang berkaitan dengan palm vein recognition, Convolutional Neural Network, Neural Architecture Search, P-DARTS, hardware-aware NAS, Knowledge Distillation, model quantization, serta evaluasi model pada perangkat edge.

Bab 3 menjelaskan metodologi penelitian, meliputi dataset dan preprocessing, pembagian data, training teacher model, proses hardware-aware P-DARTS, penyusunan latency lookup table Raspberry Pi, retraining kandidat NAS, Knowledge Distillation, export ONNX, PTQ INT8, dan skenario evaluasi.

Bab 4 menyajikan hasil eksperimen dan pembahasan, termasuk hasil search NAS, hasil retraining kandidat, hasil training teacher, hasil Knowledge Distillation, hasil quantization, serta benchmark Raspberry Pi.

Bab 5 berisi kesimpulan dari hasil penelitian serta saran untuk pengembangan penelitian selanjutnya.

---

# Catatan Integrasi ke Dokumen Lama

## Bagian yang Dipertahankan

- Narasi umum tentang pentingnya palm vein recognition.
- Alasan penggunaan deep learning/CNN untuk ekstraksi fitur otomatis.
- Fokus pada model ringan untuk perangkat edge.
- Knowledge Distillation sebagai strategi peningkatan performa student.
- Quantization sebagai bagian dari optimasi deployment.

## Bagian yang Perlu Diganti

- Jika Bab 1 lama menyebut NAS secara umum, ubah menjadi **P-DARTS**.
- Jika Bab 1 lama menyebut efisiensi hanya berdasarkan parameter atau FLOPs, tambahkan argumen bahwa latency aktual perlu diukur pada perangkat target.
- Jika Bab 1 lama menyebut pruning sebagai metode utama, hapus atau pindahkan ke saran/future work.
- Jika Bab 1 lama belum menyebut Raspberry Pi secara spesifik, tambahkan Raspberry Pi 5 sebagai target evaluasi deployment.

## Istilah yang Harus Konsisten

- Gunakan `hardware-aware NAS`, bukan berganti-ganti dengan `hardware friendly NAS`.
- Gunakan `latency lookup table` atau `LUT latency`; pilih salah satu istilah utama dan konsisten.
- Gunakan `P-DARTS` setelah pertama kali ditulis lengkap sebagai `Progressive Differentiable Architecture Search`.
- Gunakan `Post-Training Quantization (PTQ) INT8` pada penyebutan pertama.
- Gunakan `Raspberry Pi 5` untuk target perangkat, dan jelaskan bahwa perangkat ini digunakan untuk benchmark inferensi.

## Referensi yang Perlu Dimasukkan Minimal

Referensi berikut sebaiknya masuk di Bab 1 sebagai pengantar dan diperluas di Bab 2:

1. Liu et al. - DARTS: Differentiable Architecture Search.
2. Chen et al. - Progressive Differentiable Architecture Search.
3. Tan et al. - MnasNet: Platform-Aware Neural Architecture Search for Mobile.
4. Cai et al. - ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware.
5. Wu et al. - FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable NAS.
6. Howard et al. - Searching for MobileNetV3.
7. Hinton et al. - Distilling the Knowledge in a Neural Network.
8. Jacob et al. - Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.

Catatan: klaim indeks Q1/Q2 sebaiknya tidak ditulis langsung di Bab 1. Status jurnal/konferensi cukup disiapkan untuk kebutuhan daftar pustaka atau justifikasi literatur di Bab 2.

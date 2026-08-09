# Audit Kesiapan IJCCE dan Pembatasan Ruang Lingkup Eksperimen

**Topik:** *Hardware-aware* P-DARTS, *knowledge distillation*, dan kuantisasi INT8 untuk pengenalan vena telapak tangan pada Raspberry Pi  
**Target jurnal:** *International Journal of Cognitive Computing in Engineering* (IJCCE)  
**Status keputusan:** layak secara ruang lingkup, tetapi belum siap dikirim tanpa perbaikan metodologis prioritas  
**Tanggal audit:** 9 Agustus 2026

## 1. Kesimpulan Utama

IJCCE relevan dengan penelitian ini karena kontribusi yang ditawarkan berada pada persilangan *cognitive computing*, pengenalan pola biometrik, optimasi arsitektur, dan implementasi AI pada perangkat *edge*. Meskipun demikian, kesesuaian topik tidak identik dengan kesiapan naskah.

Rancangan saat ini belum dapat dinyatakan “100% memenuhi” atau pasti diterima. Klaim seperti itu tidak dapat diberikan untuk jurnal ilmiah karena editor tetap menilai ruang lingkup, orisinalitas, kualitas desain penelitian, serta kecukupan bukti. Panduan resmi IJCCE menyebutkan bahwa naskah dapat mengalami *desk rejection* karena berada di luar ruang lingkup, memiliki kemiripan tinggi, kurang orisinal, atau mengandung kelemahan dalam desain penelitian maupun metode. Naskah yang lolos pemeriksaan awal umumnya dikirim kepada sekurang-kurangnya dua reviewer independen.

Sumber resmi: [IJCCE Guide for Authors](https://www.keaipublishing.com/cn/journals/international-journal-of-cognitive-computing-in-engineering/guide-for-authors/).

Keputusan yang paling proporsional adalah **conditional go**:

- topik dan arah kontribusi layak diteruskan untuk IJCCE;
- eksperimen tidak perlu diperluas tanpa batas;
- tiga masalah yang telah diidentifikasi harus ditangani dengan urutan prioritas;
- penambahan PolyU tidak berarti seluruh NAS, seluruh *baseline*, KD, dan pengujian Raspberry Pi harus diulang dari awal.

## 2. Mengapa Rekomendasi Sebelumnya Terlihat Sangat Luas

Tiga catatan sebelumnya mempunyai tingkat urgensi yang berbeda:

1. perbedaan protokol dengan AMPVNet memengaruhi validitas perbandingan SOTA;
2. pembagian data yang berpotensi terlalu mudah memengaruhi validitas internal hasil;
3. penggunaan satu dataset memengaruhi keluasan generalisasi.

Ketiganya bukan tiga kewajiban yang harus langsung diubah menjadi satu eksperimen raksasa. Urutan logisnya adalah:

1. **audit dulu apakah split memang bermasalah;**
2. **perbaiki cara membandingkan dan cara menyatakan klaim;**
3. **tambahkan PolyU secara terbatas untuk menguji generalisasi arsitektur;**
4. perluas KD pada PolyU hanya jika hasil awal dan waktu komputasi mendukung.

Dengan pendekatan bertahap ini, keputusan eksperimen dibuat berdasarkan risiko ilmiah, bukan berdasarkan keinginan untuk menjalankan semua kombinasi yang mungkin.

## 3. Diagnosis terhadap Data dan Protokol Saat Ini

### 3.1 Fakta yang sudah dapat diverifikasi dari repositori

Konfigurasi benchmark lokal saat ini menggunakan:

- 834 kelas;
- 8 citra latih, 1 citra validasi, dan 1 citra uji per kelas;
- total 8.340 citra, terdiri atas 6.672 citra latih, 834 validasi, dan 834 uji;
- pembagian citra secara individual, misalnya `1_7.bmp`, `1_3.bmp`, dan seterusnya;
- pemeriksaan *overlap* pada tingkat nama berkas.

Sementara itu, katalog dataset IAPR TC4 mencatat SCUT_PV_v1 memiliki 11.000 citra dari 1.100 telapak milik 550 subjek. Dengan demikian, benchmark lokal menggunakan subset 834 dari 1.100 identitas telapak yang tersedia pada dataset penuh. Perbedaan ini tidak otomatis membuat eksperimen tidak sah, tetapi alasan eksklusi 266 kelas dan 2.660 citra harus dijelaskan. Jika subset terbentuk karena kegagalan praproses, kualitas ROI, berkas hilang, atau kriteria tertentu, kriteria tersebut harus bersifat objektif dan dilaporkan.

Sumber dataset: [IAPR TC4 Palm Vein Datasets](https://iapr-tc4.org/palm-vein-datasets/).

### 3.2 Hal yang belum terbukti

Belum ada bukti yang cukup untuk menyatakan bahwa split 8/1/1 saat ini pasti mengalami kebocoran data. Pemeriksaan nama berkas hanya membuktikan tidak adanya berkas identik pada dua subset. Pemeriksaan tersebut belum membuktikan bahwa citra latih dan uji tidak berasal dari urutan video, sesi, atau akuisisi yang sangat berdekatan.

Oleh sebab itu, kalimat yang tepat adalah:

> Split saat ini **berpotensi** menghasilkan evaluasi yang terlalu mudah apabila beberapa citra dari satu urutan akuisisi yang sama tersebar ke data latih, validasi, dan uji. Potensi ini harus diperiksa melalui metadata akuisisi atau protokol resmi sebelum diputuskan perlu tidaknya pelatihan ulang.

## 4. Masalah 1 — Protokol Berbeda dari AMPVNet

AMPVNet diperkenalkan sebagai model autentikasi pada SCUT_PV_v1. Penekanannya adalah pembelajaran representasi biometrik dan evaluasi kecocokan identitas, sedangkan eksperimen lokal saat ini merupakan klasifikasi *closed-set* 834 kelas. Akurasi klasifikasi dan EER/TAR pada autentikasi mengukur pertanyaan yang berbeda. Karena itu, angka dari kedua protokol tidak boleh ditempatkan dalam satu tabel seolah-olah setara.

Sumber utama: [Palm Vein Recognition under Unconstrained and Weak-Cooperative Conditions](https://doi.org/10.1109/TIFS.2024.3378427).

### 4.1 Perbaikan minimal yang wajib

- Pertahankan akurasi klasifikasi sebagai metrik utama jika tujuan penelitian memang identifikasi *closed-set* dan implementasi pada perangkat *edge*.
- Jalankan AMPVNet resmi atau implementasi yang dapat diverifikasi menggunakan split klasifikasi yang sama apabila AMPVNet dimasukkan ke tabel akurasi terkontrol.
- Jangan membandingkan akurasi 834 kelas secara langsung dengan EER yang dilaporkan paper AMPVNet.
- Ubah istilah “mengungguli SOTA” menjadi “mengungguli model pembanding pada protokol klasifikasi yang digunakan” apabila protokol tidak identik.

### 4.2 Penguat yang tidak membutuhkan pelatihan ulang

Metrik verifikasi dapat dihitung dari *checkpoint* yang sudah dilatih dengan mengekstraksi fitur sebelum *classifier*, lalu menghitung skor kemiripan. Eksperimen ini memerlukan kode evaluasi dan pembentukan pasangan atau templat, tetapi tidak selalu memerlukan pelatihan ulang model.

Skema yang dapat digunakan:

- citra latih membentuk templat pendaftaran setiap kelas;
- citra validasi digunakan untuk memilih ambang operasional;
- citra uji menjadi kueri;
- kemiripan kosinus atau jarak Euclidean menghasilkan skor *genuine* dan *impostor*;
- laporkan ROC, EER, serta TAR pada FAR yang ditentukan sebelumnya.

Hasil tersebut tetap harus disebut sebagai protokol verifikasi internal penelitian. Hasil baru boleh dibandingkan langsung dengan AMPVNet apabila pembentukan identitas, pasangan, sesi, dan metriknya disamakan dengan protokol AMPVNet.

## 5. Masalah 2 — Split Berpotensi Terlalu Mudah

Ini merupakan risiko paling penting karena dapat memengaruhi nilai akurasi 99% ke atas. Namun, langkah pertama bukan melatih ulang semua model. Langkah pertama adalah audit data.

### 5.1 Gate A — audit tanpa pelatihan

Periksa dan dokumentasikan:

1. sumber asli 10 citra pada setiap kelas;
2. apakah citra tersebut berasal dari satu atau beberapa sesi;
3. apakah beberapa citra merupakan *frame* berdekatan dari video atau gerakan yang sama;
4. apakah dataset menyediakan ID sesi, ID urutan, waktu akuisisi, atau protokol pembagian resmi;
5. alasan hanya 834 dari 1.100 telapak yang digunakan;
6. distribusi citra gagal praproses per kelas dan apakah eksklusi berkorelasi dengan kualitas citra.

### 5.2 Keputusan setelah audit

**Jika 10 citra merupakan sampel independen dan protokol resmi mendukung pembagian per citra:**

- split 8/1/1 dapat dipertahankan;
- jelaskan sumber dan dasar pembagiannya;
- tambahkan keterbatasan bahwa evaluasi tetap bersifat *closed-set* dan dalam satu dataset.

**Jika beberapa citra berasal dari sesi atau urutan yang sama:**

- buat split berbasis grup/sesi agar satu grup akuisisi tidak tersebar ke lebih dari satu subset;
- jangan langsung mengulang semua model;
- terlebih dahulu latih P-DARTS L0.05 dan dua *baseline* representatif dengan tiga *seed*;
- jika kesimpulan utama tetap konsisten, perluasan ke model lain dapat dihentikan atau dibatasi;
- jika urutan peringkat berubah drastis, tabel utama memang perlu diperbarui.

### 5.3 Kapan NAS perlu dicari ulang

Pencarian NAS tidak otomatis harus diulang hanya karena dibuat evaluasi tambahan. Arsitektur L0.05 dapat dibekukan dan diperlakukan sebagai arsitektur yang dikembangkan pada SCUT. Pencarian ulang baru menjadi wajib apabila:

- split lama diketahui memakai data uji untuk seleksi arsitektur atau hiperparameter;
- validasi selama *search* terbukti bocor pada tingkat sesi/urutan;
- naskah mengklaim bahwa arsitektur tersebut adalah hasil optimal untuk split baru.

Jika tidak ada kebocoran terhadap data uji, pilihan yang lebih hemat adalah mempertahankan *genotype*, melatih ulang bobot pada split yang diperbaiki, dan menyebutnya sebagai validasi arsitektur tetap.

## 6. Masalah 3 — Apakah Satu Dataset Tidak Cukup?

Dua dataset **bukan persyaratan formal IJCCE**. Panduan jurnal tidak menyatakan bahwa setiap artikel harus menggunakan minimal dua dataset. Oleh sebab itu, penggunaan satu dataset tidak otomatis menyebabkan penolakan.

Namun, satu dataset mempersempit klaim yang dapat dibuat. Dengan SCUT saja, klaim yang aman adalah:

> Metode menghasilkan *trade-off* akurasi, ukuran, dan *latency* yang baik pada SCUT_PV_v1 dan Raspberry Pi dengan protokol yang digunakan.

Klaim berikut belum didukung oleh satu dataset:

> Arsitektur memiliki generalisasi yang konsisten untuk pengenalan vena telapak tangan pada berbagai sensor, sesi, dan kondisi akuisisi.

Karena penelitian mengusulkan arsitektur hasil NAS, satu validasi eksternal akan sangat memperkuat argumen bahwa arsitektur tidak hanya menyesuaikan karakter SCUT. PolyU dapat dipakai untuk tujuan ini tanpa menggandakan seluruh penelitian.

## 7. Cara Menambahkan PolyU tanpa Mengulang Seluruh Pipeline

PolyU Multispectral Palmprint Database memiliki 500 telapak, 6 citra per sesi, dan dua sesi dengan jarak rata-rata sekitar sembilan hari pada setiap spektrum. Untuk eksperimen vena, gunakan hanya kanal NIR dan sebutkan secara presisi bahwa data berasal dari subset NIR PolyU Multispectral Palmprint Database. Kanal dan versi dataset tidak boleh dicampur tanpa penjelasan.

Sumber karakteristik dataset:

- [NIST Biometric Research Database Catalog — PolyU Multispectral Palmprint](https://tsapps.nist.gov/BDbC/Search/Details/476)
- [Zhang et al., An Online System of Multispectral Palmprint Verification](https://doi.org/10.1109/TIM.2009.2028772)

### 7.1 Protokol PolyU yang disarankan

Gunakan pembagian berbasis sesi, bukan pengacakan seluruh citra:

- Sesi 1: data pengembangan/pelatihan;
- sebagian terkontrol dari Sesi 1: validasi dan pemilihan *checkpoint*;
- Sesi 2: data uji yang tidak disentuh selama pelatihan;
- identitas telapak kiri dan kanan diperlakukan konsisten sesuai protokol dataset;
- seluruh model memakai ROI, augmentasi, dan aturan pemilihan *checkpoint* yang sama.

Pembagian tepat data latih/validasi dari enam citra Sesi 1 harus ditetapkan sebelum eksperimen. Contoh awal adalah lima citra latih dan satu citra validasi per telapak, sedangkan seluruh enam citra Sesi 2 digunakan sebagai data uji. Skema tersebut perlu diverifikasi terhadap struktur berkas PolyU yang benar-benar dimiliki sebelum dibuat permanen.

### 7.2 Eksperimen PolyU minimum

PolyU digunakan untuk menguji **generalisasi arsitektur**, bukan mengulang penemuan arsitektur. Jalankan hanya:

| Model | Peran | Inisialisasi | KD | Seed |
|---|---|---|---|---|
| P-DARTS L0.05 | model usulan dengan *genotype* tetap | acak | tidak | 42, 123, 2026 |
| AMPVNet | pembanding spesifik vena telapak tangan | acak | tidak | 42, 123, 2026 |
| Satu *baseline* ringan terkuat dari SCUT | pembanding generik | acak | tidak | 42, 123, 2026 |

Total tambahan tahap minimum adalah **9 pelatihan**, bukan seluruh kombinasi model. Tidak dilakukan:

- pencarian ulang P-DARTS pada PolyU;
- pelatihan semua *baseline* pada PolyU;
- eksperimen *pretrained* semua model pada PolyU;
- KD semua model pada PolyU;
- kuantisasi dan pengujian Raspberry Pi untuk semua model PolyU.

Hasil PolyU dilaporkan sebagai tabel validasi eksternal terpisah. Nilainya tidak dirata-ratakan dengan SCUT karena jumlah kelas, sensor, kondisi akuisisi, dan distribusi data berbeda.

## 8. Apakah *Teacher* Harus Dilatih Ulang untuk PolyU?

Jawabannya bergantung pada klaim yang ingin dibuat.

### 8.1 Jika PolyU hanya menguji generalisasi arsitektur

**Tidak perlu melatih *teacher*.** P-DARTS L0.05 dan pembanding dilatih tanpa KD pada PolyU. Eksperimen ini menjawab:

> Apakah arsitektur hasil pencarian pada SCUT tetap kompetitif ketika bobotnya dilatih pada dataset dan sesi akuisisi yang berbeda?

Ini merupakan pilihan minimum yang paling hemat dan tetap berguna secara ilmiah.

### 8.2 Jika ingin mengklaim manfaat KD juga berlaku pada PolyU

**Ya, diperlukan *teacher* yang dilatih atau disesuaikan pada PolyU.** *Teacher* SCUT berbasis logit tidak dapat langsung mendistilasi kelas PolyU karena ruang kelas dan *classifier* berbeda.

Namun, *teacher* tidak perlu dilatih ulang untuk setiap model atau setiap *seed student*. Pilihan hemat yang masih dapat dipertanggungjawabkan adalah:

1. latih satu *teacher* PolyU dengan konfigurasi yang ditetapkan sebelumnya;
2. bekukan satu *checkpoint teacher* terbaik berdasarkan validasi;
3. gunakan *teacher* yang sama untuk tiga *seed student* P-DARTS;
4. bandingkan P-DARTS tanpa KD dan dengan KD pada tiga *seed* yang sama.

Tambahan tahap ini adalah satu pelatihan *teacher* dan tiga pelatihan *student* KD. Keterbatasannya harus dinyatakan: variasi yang diukur berasal dari inisialisasi *student*, bukan variasi pelatihan *teacher*. Jika sumber daya memadai, tiga *teacher seed* merupakan desain yang lebih kuat, tetapi bukan langkah pertama yang wajib.

### 8.3 Jika ingin mengklaim pipeline NAS + KD sepenuhnya lintas-dataset

Baru pada klaim ini *teacher*, ablation KD, dan evaluasi model usulan perlu dijalankan pada kedua dataset. Pencarian NAS tetap tidak perlu diulang di PolyU selama klaimnya adalah transferabilitas arsitektur yang ditemukan di SCUT, bukan penemuan arsitektur optimal khusus PolyU.

## 9. Rencana Eksperimen Berjenjang yang Dibatasi

### Tahap 0 — wajib sebelum memakai GPU tambahan

1. Audit asal 834 kelas dan alasan tidak memakai 1.100 kelas penuh.
2. Audit sesi/urutan akuisisi pada split SCUT.
3. Pastikan data uji tidak pernah digunakan untuk memilih *genotype*, lambda, epoch, *checkpoint*, atau hiperparameter.
4. Tetapkan klaim utama: identifikasi *closed-set*, verifikasi biometrik, atau keduanya.

**Keluaran:** laporan audit data dan keputusan apakah split lama dipertahankan atau diperbaiki.

### Tahap 1 — inti SCUT yang wajib

Gunakan empat kelompok model yang representatif:

1. P-DARTS L0.05 tanpa KD;
2. MobileNetV3Small sebagai arsitektur ringan manual;
3. ProxylessNAS-Mobile sebagai pembanding NAS yang berorientasi perangkat;
4. AMPVNet resmi sebagai pembanding spesifik vena telapak tangan.

Jalankan protokol *scratch* terkontrol dengan tiga *seed*. Tambahkan ShuffleNetV2, FBNet-C, MnasNet-A1, EfficientNetLite0, dan rekonstruksi Ding hanya jika diperlukan untuk memperluas cakupan, bukan sebagai syarat untuk membuktikan setiap klaim.

### Tahap 2 — kontribusi KD pada SCUT

Terapkan *teacher*, suhu, bobot loss, split, dan anggaran pelatihan yang sama pada:

- P-DARTS L0.05;
- satu atau dua *baseline* ringan terkuat dari Tahap 1.

Tahap ini memisahkan pengaruh arsitektur dari pengaruh KD. Tidak perlu memberi KD kepada semua model.

### Tahap 3 — implementasi Raspberry Pi

Untuk P-DARTS dan model pembanding utama:

- ekspor FP32 dan INT8;
- evaluasi kembali akurasi setelah kuantisasi;
- ukur *latency* dengan jumlah *warm-up*, iterasi, *thread*, *governor*, dan kondisi termal yang sama;
- laporkan median dan p95, bukan hanya rata-rata;
- pisahkan hasil PyTorch, ONNX, dan ONNX Runtime Raspberry Pi.

### Tahap 4 — validasi eksternal PolyU minimum

Jalankan sembilan pelatihan pada tiga model sebagaimana Tabel Bagian 7.2. Hentikan pada tahap ini jika tujuan yang ingin dibuktikan hanya generalisasi arsitektur.

### Tahap 5 — PolyU + KD bersyarat

Tahap ini dijalankan hanya apabila:

- P-DARTS tetap kompetitif pada PolyU tanpa KD;
- waktu komputasi masih tersedia;
- naskah akan mengklaim bahwa manfaat KD juga konsisten lintas-dataset.

## 10. Matriks Wajib, Disarankan, dan Opsional

| Komponen | Status | Alasan |
|---|---|---|
| Audit asal subset 834/1.100 kelas | Wajib | Menentukan risiko bias seleksi data |
| Audit sesi/urutan split SCUT | Wajib | Menentukan validitas internal akurasi |
| Tiga *seed* untuk eksperimen utama | Wajib | Mengukur variasi inisialisasi |
| Pemisahan tabel *scratch*, *pretrained*, KD, dan INT8 | Wajib | Menghindari perbandingan dengan perlakuan berbeda |
| AMPVNet pada protokol lokal atau pembatasan klaim | Wajib | Mencegah perbandingan metrik yang tidak setara |
| Metrik verifikasi dari *checkpoint* yang ada | Sangat disarankan | Memperkuat relevansi biometrik dengan biaya rendah |
| PolyU dengan *genotype* tetap dan tanpa KD | Sangat disarankan | Menguji generalisasi arsitektur dengan tambahan terbatas |
| Satu *teacher* PolyU + KD hanya pada P-DARTS | Opsional bersyarat | Diperlukan hanya untuk klaim KD lintas-dataset |
| Pencarian NAS ulang pada PolyU | Tidak diperlukan | Mengubah tujuan dari transfer arsitektur menjadi pencarian per-dataset |
| Semua *baseline* dilatih pada PolyU | Tidak diperlukan | Biaya besar dengan tambahan bukti yang semakin kecil |
| Semua model PolyU diuji pada Raspberry Pi | Tidak diperlukan | *Deployment* utama dapat difokuskan pada SCUT dan model final |
| Tiga *teacher seed* pada setiap dataset | Penguat lanjutan | Lebih kuat secara statistik, tetapi bukan prioritas awal |

## 11. Bentuk Tabel Hasil yang Disarankan

Hasil tidak digabung menjadi satu tabel besar. Gunakan struktur berikut:

1. **Tabel A — Controlled SCUT architecture comparison**  
   Semua model dari inisialisasi acak, tanpa KD, tiga *seed*, dan resep yang sama.

2. **Tabel B — Pretrained practical baselines**  
   Hanya model dengan bobot resmi; dilaporkan terpisah dari Tabel A.

3. **Tabel C — KD ablation on SCUT**  
   P-DARTS dan satu atau dua *baseline* terkuat, masing-masing tanpa dan dengan KD.

4. **Tabel D — FP32/INT8 Raspberry Pi deployment**  
   Akurasi, ukuran, parameter, FLOPs, median *latency*, p95, dan memori jika tersedia.

5. **Tabel E — PolyU session-based external validation**  
   P-DARTS, AMPVNet, dan satu *baseline* ringan, tanpa KD pada tahap minimum.

6. **Tabel F — PolyU KD ablation**  
   Hanya ditambahkan jika Tahap 5 dijalankan.

## 12. Klaim yang Aman dan Klaim yang Harus Dihindari

### Klaim yang aman setelah Tahap 0–3

> Pada protokol klasifikasi SCUT_PV_v1 yang dikendalikan, arsitektur P-DARTS L0.05 memberikan *trade-off* antara akurasi, ukuran model, dan *latency* Raspberry Pi dibandingkan model pembanding yang diuji.

> Ablation KD menunjukkan perubahan performa ketika arsitektur dan konfigurasi pelatihan lainnya dipertahankan.

### Klaim yang aman setelah PolyU minimum

> *Genotype* yang ditemukan pada SCUT tetap kompetitif ketika bobotnya dilatih ulang menggunakan pembagian berbasis sesi pada PolyU-NIR, sehingga memberikan bukti awal transferabilitas arsitektur antar-dataset.

### Klaim yang harus dihindari

- “Metode terbukti unggul secara universal.”
- “Model mengungguli AMPVNet” apabila metrik dan protokolnya berbeda.
- “Model memiliki generalisasi lintas-dataset” jika hanya diuji pada SCUT.
- “KD terbukti konsisten pada dua dataset” jika KD hanya dijalankan pada SCUT.
- “Split bebas kebocoran” jika yang diperiksa hanya duplikasi nama berkas.
- “SOTA” tanpa protokol dan metode pembanding yang benar-benar sebanding.

## 13. Simulasi Reviewer #2

### Kritik paling mungkin

> Akurasi hampir sempurna diperoleh dari hanya satu citra uji per kelas dan pembagian per citra. Penulis belum membuktikan bahwa citra dari akuisisi yang sama tidak tersebar ke data latih dan uji. Selain itu, 834 dari 1.100 kelas SCUT digunakan tanpa penjelasan. Perbandingan dengan metode autentikasi berbasis EER juga tidak setara dengan akurasi klasifikasi *closed-set*.

### Respons yang kuat

> Kami mengaudit metadata dan pengelompokan akuisisi sebelum menetapkan split. Alasan eksklusi kelas dilaporkan beserta kriteria objektifnya. Perbandingan arsitektur dilakukan pada protokol klasifikasi yang identik, sedangkan hasil autentikasi dilaporkan pada tabel terpisah. Untuk menguji transferabilitas arsitektur tanpa mengulang proses pencarian, *genotype* dibekukan dan dilatih ulang pada PolyU-NIR menggunakan pembagian berbasis sesi.

Respons tersebut hanya boleh digunakan setelah semua langkah yang disebutkan benar-benar dilakukan.

## 14. Keputusan Akhir tentang Beban Eksperimen

Penelitian tidak perlu dirombak menjadi seluruh kombinasi berikut:

```text
2 dataset × semua model × scratch × pretrained × KD × INT8 × 3 seed × NAS ulang
```

Ruang lingkup yang direkomendasikan adalah:

```text
SCUT:
  pipeline utama lengkap + baseline representatif + KD fairness + deployment

PolyU:
  genotype tetap + 2 baseline + 3 seed + split berbasis sesi
  KD hanya jika klaim lintas-dataset memang dibutuhkan
```

Prioritas ilmiah tertinggi bukan menambah jumlah pelatihan, melainkan memastikan bahwa data uji benar-benar independen, protokol pembanding setara, dan setiap klaim sesuai dengan bukti. Setelah audit split selesai, kebutuhan pelatihan ulang dapat diputuskan secara objektif. Sampai audit tersebut selesai, menjalankan puluhan eksperimen tambahan berisiko menghasilkan banyak angka tanpa menyelesaikan kelemahan metodologis utama.

## 15. Checklist Keputusan Berikutnya

- [ ] Jelaskan mengapa dataset lokal berisi 834, bukan 1.100 telapak.
- [ ] Temukan metadata sesi/urutan atau konfirmasi protokol 10 citra SCUT.
- [ ] Putuskan apakah split 8/1/1 dipertahankan atau diganti berbasis grup.
- [ ] Tetapkan AMPVNet sebagai pembanding klasifikasi lokal, pembanding verifikasi, atau keduanya.
- [ ] Pilih satu *baseline* ringan terkuat untuk validasi PolyU.
- [ ] Verifikasi bahwa dataset PolyU yang dimiliki memuat kanal NIR dan dua sesi.
- [ ] Bekukan protokol PolyU sebelum pelatihan pertama.
- [ ] Jalankan PolyU tanpa KD terlebih dahulu.
- [ ] Putuskan PolyU + KD berdasarkan hasil tahap minimum dan klaim artikel.
- [ ] Tulis keterbatasan yang tersisa secara eksplisit.

---

### Catatan integritas ilmiah

Dokumen ini merupakan audit desain dan rencana pengambilan keputusan, bukan laporan hasil. Tidak ada nilai performa PolyU, EER, atau signifikansi statistik yang diasumsikan sebelum eksperimen dilakukan. Setiap klaim akhir harus mengikuti data aktual, konfigurasi yang tersimpan, dan protokol yang benar-benar dijalankan.

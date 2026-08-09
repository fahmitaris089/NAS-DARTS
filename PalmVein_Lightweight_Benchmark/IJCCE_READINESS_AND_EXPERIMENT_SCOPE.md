# Audit Kesiapan IJCCE dan Pembatasan Ruang Lingkup Eksperimen

**Topik:** *Hardware-aware* P-DARTS, *knowledge distillation*, dan kuantisasi INT8 untuk pengenalan vena telapak tangan pada Raspberry Pi  
**Target jurnal:** *International Journal of Cognitive Computing in Engineering* (IJCCE)  
**Status keputusan:** layak secara ruang lingkup, tetapi belum siap dikirim tanpa perbaikan metodologis prioritas  
**Ruang lingkup aktif:** perbandingan *scratch*, perbandingan *pretrained*, dan perbandingan KD yang adil pada subset SCUT 834 kelas  
**Ruang lingkup ditunda:** validasi eksternal PolyU sampai tiga eksperimen inti SCUT selesai  
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

Untuk pelaksanaan terdekat, penelitian difokuskan pada tiga eksperimen inti SCUT: perbandingan arsitektur dari inisialisasi acak, perbandingan praktis menggunakan bobot *pretrained*, dan evaluasi KD yang dikendalikan. PolyU dicatat sebagai validasi eksternal tahap berikutnya, tetapi belum menjadi pekerjaan aktif.

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

Sementara itu, katalog dataset IAPR TC4 mencatat SCUT_PV_v1 memiliki 11.000 citra dari 1.100 telapak milik 550 subjek. Dataset yang digunakan dalam penelitian ini bukan hasil pemilihan 834 kelas oleh peneliti, melainkan subset 834 kelas yang diberikan langsung oleh pihak pemilik penelitian/dataset. Dengan demikian, penggunaan subset tersebut dapat dipertahankan selama asal data, izin penggunaan, jumlah kelas, dan jumlah citra dilaporkan secara transparan.

Belum ada konfirmasi tertulis bahwa 266 kelas lainnya tidak dapat dipublikasikan. Oleh sebab itu, dugaan mengenai pembatasan publikasi tidak boleh dinyatakan sebagai fakta. Formulasi yang aman untuk naskah adalah:

> Penelitian menggunakan subset SCUT_PV_v1 yang terdiri atas 834 identitas telapak dan 8.340 citra. Subset tersebut disediakan langsung oleh pihak pemilik dataset untuk keperluan penelitian ini; pemilihan atau eksklusi kelas tidak dilakukan oleh peneliti.

Jika pemilik memberikan keterangan tertulis mengenai alasan hanya 834 kelas yang dibagikan, informasi tersebut dapat ditambahkan pada bagian dataset atau *data availability statement*. Bukti komunikasi dan izin penggunaan sebaiknya diarsipkan, tetapi isi korespondensi privat tidak perlu dipublikasikan tanpa persetujuan.

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

## 5. Masalah 2 — Keadilan dan Kekuatan Split 8/1/1

Split 8/1/1 mempunyai dua aspek yang harus dibedakan.

1. **Keadilan komparatif:** split ini adil untuk membandingkan arsitektur apabila seluruh model menggunakan daftar citra latih, validasi, dan uji yang sama. *Seed* 42, 123, dan 2026 harus mengubah inisialisasi serta urutan pelatihan, bukan membuat split baru untuk setiap model.
2. **Kekuatan evaluasi biometrik:** satu citra validasi dan satu citra uji per kelas membatasi pengukuran variasi intra-kelas. Dengan 834 citra uji, satu prediksi mengubah akurasi sekitar 0,12 poin persentase. Akurasi 99,64% setara dengan sekitar tiga kesalahan, sehingga selisih yang sangat kecil antarmodel perlu dibaca bersama simpangan baku tiga *seed* dan jumlah kesalahan absolut.

Kesimpulannya, **split 8/1/1 tidak perlu diubah hanya karena rasionya**. Mengubah split setelah hasil awal diketahui akan mengubah protokol penelitian dan mewajibkan pelatihan ulang seluruh model yang ingin dibandingkan. Perubahan baru diperlukan apabila ditemukan bukti kebocoran berbasis sesi/urutan atau ketidaksesuaian serius dengan struktur akuisisi dataset.

### 5.1 Gate A — audit tanpa pelatihan

Periksa dan dokumentasikan:

1. sumber asli 10 citra pada setiap kelas;
2. apakah citra tersebut berasal dari satu atau beberapa sesi;
3. apakah beberapa citra merupakan *frame* berdekatan dari video atau gerakan yang sama;
4. apakah dataset menyediakan ID sesi, ID urutan, waktu akuisisi, atau protokol pembagian resmi;
5. bukti bahwa subset 834 kelas memang diterima langsung dari pemilik dataset;
6. izin penggunaan dan batas distribusi subset;
7. apakah pemilik dapat mengonfirmasi struktur sesi/urutan dari 10 citra per kelas.

### 5.2 Keputusan setelah audit

**Jika 10 citra merupakan sampel independen dan protokol resmi mendukung pembagian per citra:**

- split 8/1/1 dapat dipertahankan;
- jelaskan sumber dan dasar pembagiannya;
- tambahkan keterbatasan bahwa evaluasi tetap bersifat *closed-set* dan dalam satu dataset.

**Jika informasi sesi/urutan tidak tersedia, tetapi tidak ada bukti kebocoran:**

- pertahankan split 8/1/1 sebagai protokol utama yang telah ditetapkan;
- gunakan hash split yang sama untuk seluruh model dan *seed*;
- nyatakan bahwa independensi pada tingkat sesi belum dapat diverifikasi;
- hindari klaim generalisasi sesi atau kondisi akuisisi;
- gunakan PolyU berbasis sesi pada tahap berikutnya sebagai validasi eksternal.

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

### 5.4 Keputusan operasional split

Keputusan untuk eksperimen aktif adalah:

```text
Split utama       : tetap 8 train / 1 validation / 1 test per kelas
Jumlah kelas      : 834 kelas yang diberikan pemilik dataset
Split antar-seed  : tetap sama; tidak diacak ulang
Pemilihan model   : minimum validation loss
Penggunaan test   : satu kali setelah checkpoint dipilih
Pelaporan         : mean ± sample standard deviation dari 3 seed
Catatan wajib     : satu citra uji per kelas dan status sesi/urutan
```

Keputusan ini membekukan protokol sebelum eksperimen pembanding tambahan dijalankan. Jika kelak ditemukan bukti kebocoran sesi/urutan, hasil dengan split lama dipertahankan sebagai hasil awal dan eksperimen koreksi dilaporkan terpisah; hasil lama tidak boleh diam-diam diganti.

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

## 9. Ruang Lingkup Aktif: Tiga Eksperimen Inti SCUT

### 9.1 Prasyarat sebelum menjalankan batch eksperimen

1. Catat bahwa 834 kelas merupakan subset yang diberikan langsung oleh pemilik dataset.
2. Arsipkan izin penggunaan atau korespondensi penyerahan dataset.
3. Gunakan satu file split 8/1/1 dan satu hash split untuk semua eksperimen.
4. Pastikan data uji tidak digunakan untuk memilih *genotype*, lambda, *checkpoint*, hiperparameter, atau dua *baseline* penerima KD.
5. Validasi implementasi, jumlah parameter, dan bentuk keluaran seluruh model sebelum pelatihan panjang.

**Keluaran:** protokol dan daftar model dibekukan sebelum batch pelatihan dimulai.

### 9.2 Eksperimen 1 — perbandingan arsitektur terkontrol

Semua model dilatih dari inisialisasi acak selama maksimum 600 epoch dengan *seed* 42, 123, dan 2026:

1. P-DARTS L0.05 tanpa KD;
2. MobileNetV3Small;
3. ShuffleNetV2 x0.5;
4. ShuffleNetV2 x1.0;
5. EfficientNetLite0;
6. ProxylessNAS-Mobile;
7. FBNet-C;
8. MnasNet-A1;
9. satu model Ding utama yang didefinisikan sebelum eksperimen;
10. AMPVNet jika implementasi resmi atau implementasi tervalidasi telah tersedia.

Semua model harus menggunakan:

- inisialisasi acak;
- split 8/1/1 yang identik;
- praproses dan augmentasi yang identik;
- optimizer, *scheduler*, *learning rate*, `weight_decay`, dan `batch_size` yang identik;
- batas maksimum 600 epoch;
- aturan pemilihan *checkpoint* berdasarkan minimum *validation loss*;
- evaluasi data uji satu kali setelah *checkpoint* dipilih;
- tanpa KD dan tanpa bobot ImageNet.

Jumlah epoch diperlakukan sebagai **anggaran maksimum yang sama**, bukan bukti bahwa setiap arsitektur membutuhkan tepat 600 epoch untuk konvergen. Seluruh kurva pelatihan harus disimpan. Tabel ini menjawab:

> Pada resep pelatihan dan anggaran komputasi yang dikendalikan, apakah P-DARTS L0.05 menawarkan *trade-off* akurasi dan efisiensi yang lebih baik dibandingkan arsitektur pembanding?

Catatan untuk Ding: `ding_baseline`, `ding_pw`, dan `ding_pruned` tidak boleh digabung menjadi label “Ding” tanpa penjelasan. Pilih satu arsitektur utama sebelum melihat hasil uji. Varian lainnya ditempatkan sebagai rekonstruksi/ablation tambahan. Jika kode bukan implementasi resmi penulis, gunakan label “independent reconstruction based on Ding et al.”

Catatan untuk AMPVNet: model hanya boleh disebut implementasi resmi jika kode atau bobot benar-benar berasal dari penulis. Jika implementasi direkonstruksi, provenance dan deviasi arsitektur harus dijelaskan.

### 9.3 Eksperimen 2 — perbandingan praktis dengan *pretrained baseline*

Model yang memiliki bobot ImageNet publik dan dapat diverifikasi dilatih menggunakan *fine-tuning* selama maksimum 200 epoch dengan tiga *seed*:

1. MobileNetV3Small;
2. ShuffleNetV2 x0.5 dan x1.0, apabila bobot untuk kedua varian tersedia dan dapat diverifikasi;
3. EfficientNetLite0;
4. ProxylessNAS-Mobile;
5. FBNet-C;
6. MnasNet-A1;
7. EfficientNetV2M sebagai *teacher*/referensi kapasitas besar.

Tabel *pretrained* harus dipisahkan dari tabel *scratch*. P-DARTS L0.05 dapat ditampilkan sebagai baris referensi dari hasil *scratch* atau KD, tetapi tidak boleh diberi label *pretrained* karena tidak memiliki bobot ImageNet yang setara.

Frasa “bobot resmi” hanya digunakan jika bobot diterbitkan oleh pemilik model atau repositori resmi. Bobot dari `timm`, Torch Hub, atau repositori pihak ketiga tetap dapat digunakan, tetapi sumber, nama *checkpoint*, versi pustaka, resolusi input, dan normalisasi harus dicatat. Jika bobot publik yang dapat diverifikasi tidak tersedia untuk arsitektur yang tepat, model tersebut dikeluarkan dari tabel ini.

Perbedaan 600 epoch pada *scratch* dan 200 epoch pada *pretrained* dapat diterima karena kedua tabel menjawab pertanyaan berbeda. Aturan minimum *validation loss* tetap digunakan agar epoch 200 tidak otomatis dianggap sebagai *checkpoint* terbaik. Tabel ini menjawab:

> Apakah model usulan tetap kompetitif terhadap penggunaan praktis arsitektur yang memanfaatkan bobot pra-pelatihan publik?

EfficientNetV2M harus diberi kategori “*teacher*/capacity reference”, bukan “*lightweight baseline*”, agar pembaca tidak menafsirkan perbandingannya sebagai perbandingan perangkat yang setara.

### 9.4 Eksperimen 3 — perbandingan KD yang adil

KD diterapkan pada:

- P-DARTS L0.05;
- dua *baseline* ringan yang dipilih dari Eksperimen 1 menggunakan hasil validasi dan kriteria Pareto yang ditetapkan sebelumnya.

Contoh kandidat awal ialah MobileNetV3Small dan ShuffleNetV2 x1.0 atau ProxylessNAS-Mobile. Pemilihan final tidak boleh didasarkan pada data uji. Kriteria yang disarankan adalah satu model dengan akurasi validasi tertinggi dan satu model dengan *trade-off* akurasi–ukuran terbaik di bawah batas ukuran yang ditentukan sebelum evaluasi data uji.

Seluruh *student* KD menggunakan:

- *teacher checkpoint* yang sama;
- *temperature*, `alpha`, formulasi loss, dan normalisasi logit yang sama;
- split dan augmentasi yang sama;
- maksimum 600 epoch dan aturan *checkpoint* yang sama;
- tiga *seed student* yang sama: 42, 123, dan 2026.

Perbandingan utama harus dilakukan **di dalam arsitektur yang sama**:

```text
P-DARTS tanpa KD       vs P-DARTS + KD
Baseline A tanpa KD    vs Baseline A + KD
Baseline B tanpa KD    vs Baseline B + KD
```

Desain ini memisahkan peningkatan dari KD dan perbedaan yang berasal dari arsitektur. Satu *teacher* tetap dapat digunakan untuk seluruh *student seed* asalkan keputusan tersebut dilaporkan; simpangan baku kemudian merepresentasikan variasi *student*, bukan variasi *teacher*.

### 9.5 Deployment setelah tiga eksperimen inti

Pernyataan “seluruh model menjadi INT8” harus dipersempit menjadi seluruh model yang masuk tabel deployment final. Minimum yang disarankan:

- seluruh model pada Eksperimen 1;
- P-DARTS dan dua *baseline* pada Eksperimen 3;
- model *pretrained* hanya jika tabel deployment juga mengklaim skenario praktis *pretrained*.

Untuk setiap model, akurasi FP32 dan INT8 harus berasal dari pipeline yang diberi label jelas. Kuantisasi semua *seed* diperlukan jika akurasi INT8 akan dilaporkan sebagai mean ± simpangan baku. Pengukuran *latency* Raspberry Pi harus menggunakan konfigurasi perangkat, *thread*, *warm-up*, jumlah iterasi, dan kondisi termal yang identik.

### 9.6 Status PolyU selama scope aktif

PolyU **ditunda**, bukan dibatalkan. Eksperimen PolyU dimulai setelah tabel *scratch*, *pretrained*, dan KD SCUT selesai serta dua *baseline* terkuat telah ditentukan. Ruang lingkup minimum PolyU tetap:

- P-DARTS L0.05;
- AMPVNet;
- satu *baseline* ringan terkuat dari SCUT;
- tiga *seed*;
- tanpa pencarian NAS ulang;
- tanpa KD pada tahap awal.

Totalnya sembilan pelatihan. *Teacher* PolyU hanya dilatih jika artikel akan mengklaim konsistensi manfaat KD lintas-dataset.

### 9.7 Estimasi beban komputasi scope aktif

Jika seluruh model tersedia, perkiraan jumlah pelatihan adalah:

| Eksperimen | Konfigurasi model | Seed | Total pelatihan |
|---|---:|---:|---:|
| *Scratch* terkontrol | 10 | 3 | 30 |
| *Pretrained* praktis, termasuk EfficientNetV2M | maksimum 8 | 3 | maksimum 24 |
| KD yang adil | 3 *student* | 3 | 9 |
| **Total maksimum** |  |  | **63** |

Angka tersebut belum menghitung ekspor ONNX, kuantisasi, evaluasi INT8, dan pengulangan *benchmark* Raspberry Pi. Jika AMPVNet belum tersedia atau suatu arsitektur tidak mempunyai bobot *pretrained* yang dapat diverifikasi, konfigurasi itu tidak boleh diganti diam-diam dengan model lain; jumlah pelatihan dikurangi dan alasannya dicatat.

Dengan demikian, tiga eksperimen inti sudah merupakan ruang lingkup yang besar, tetapi koheren. Urutan eksekusinya harus berbentuk *gate*: validasi semua model dengan *smoke test*, selesaikan tabel *scratch*, pilih penerima KD dari validasi, selesaikan tabel *pretrained*, kemudian jalankan KD. PolyU tidak dijalankan bersamaan dengan tahap ini.

## 10. Matriks Wajib, Disarankan, dan Opsional

| Komponen | Status | Alasan |
|---|---|---|
| Dokumentasi bahwa subset 834 kelas diberikan pemilik | Wajib | Menjelaskan provenance dan mencegah kesan seleksi kelas oleh peneliti |
| Konfirmasi tertulis alasan 266 kelas tidak dibagikan | Disarankan | Tidak boleh diasumsikan sebagai pembatasan publikasi tanpa bukti |
| Audit sesi/urutan split SCUT | Disarankan kuat | Menentukan batas klaim independensi data |
| Mempertahankan satu split 8/1/1 untuk semua model | Diputuskan | Menjaga perbandingan arsitektur konsisten |
| Tiga *seed* untuk eksperimen utama | Wajib | Mengukur variasi inisialisasi |
| Pemisahan tabel *scratch*, *pretrained*, KD, dan INT8 | Wajib | Menghindari perbandingan dengan perlakuan berbeda |
| AMPVNet pada protokol lokal atau pembatasan klaim | Wajib | Mencegah perbandingan metrik yang tidak setara |
| Metrik verifikasi dari *checkpoint* yang ada | Sangat disarankan | Memperkuat relevansi biometrik dengan biaya rendah |
| PolyU dengan *genotype* tetap dan tanpa KD | Ditunda | Dimulai setelah tiga eksperimen inti SCUT selesai |
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
   Hanya model dengan bobot publik yang provenance-nya dapat diverifikasi; dilaporkan terpisah dari Tabel A.

3. **Tabel C — KD ablation on SCUT**  
   P-DARTS dan satu atau dua *baseline* terkuat, masing-masing tanpa dan dengan KD.

4. **Tabel D — FP32/INT8 Raspberry Pi deployment**  
   Akurasi, ukuran, parameter, FLOPs, median *latency*, p95, dan memori jika tersedia.

5. **Tabel E — PolyU session-based external validation**  
   P-DARTS, AMPVNet, dan satu *baseline* ringan, tanpa KD pada tahap minimum.

6. **Tabel F — PolyU KD ablation**  
   Hanya ditambahkan jika eksperimen opsional pada Bagian 8.2 dijalankan.

## 12. Klaim yang Aman dan Klaim yang Harus Dihindari

### Klaim yang aman setelah tiga eksperimen inti SCUT dan deployment

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

> Akurasi hampir sempurna diperoleh dari hanya satu citra uji per kelas dan pembagian per citra. Penulis belum membuktikan bahwa citra dari akuisisi yang sama tidak tersebar ke data latih dan uji. Selain itu, provenance subset 834 dari 1.100 kelas SCUT belum dijelaskan. Perbandingan dengan metode autentikasi berbasis EER juga tidak setara dengan akurasi klasifikasi *closed-set*.

### Respons yang kuat

> Kami menggunakan subset 834 kelas yang diberikan langsung oleh pemilik dataset dan tidak melakukan eksklusi kelas secara mandiri. Split 8/1/1 serta hash split dibekukan dan digunakan secara identik untuk seluruh model. Keterbatasan satu citra uji per kelas dan status metadata sesi dinyatakan secara eksplisit. Perbandingan arsitektur dilakukan pada protokol klasifikasi yang identik, sedangkan hasil autentikasi dilaporkan terpisah. Validasi PolyU-NIR berbasis sesi ditempatkan sebagai eksperimen eksternal tahap berikutnya.

Respons tersebut hanya boleh digunakan setelah semua langkah yang disebutkan benar-benar dilakukan.

## 14. Keputusan Akhir tentang Beban Eksperimen

Penelitian tidak perlu dirombak menjadi seluruh kombinasi berikut:

```text
2 dataset × semua model × scratch × pretrained × KD × INT8 × 3 seed × NAS ulang
```

Ruang lingkup yang direkomendasikan adalah:

```text
SCUT:
  AKTIF
  1. scratch terkontrol, 600 epoch maksimum × 3 seed
  2. pretrained praktis, 200 epoch maksimum × 3 seed
  3. KD fairness, 600 epoch maksimum × 3 seed
  4. deployment FP32/INT8 setelah checkpoint final tersedia

PolyU:
  DITUNDA SAMPAI EKSPERIMEN INTI SCUT SELESAI
  genotype tetap + 2 baseline + 3 seed + split berbasis sesi
  KD hanya jika klaim lintas-dataset memang dibutuhkan
```

Prioritas ilmiah tertinggi bukan menambah jumlah pelatihan, melainkan menjaga protokol pembanding tetap setara dan memastikan setiap klaim sesuai dengan bukti. Split 8/1/1 dipertahankan untuk scope aktif, sedangkan status metadata sesi/urutan harus dikonfirmasi atau dinyatakan sebagai keterbatasan. PolyU baru diproses setelah tiga tabel inti SCUT selesai.

## 15. Checklist Keputusan Berikutnya

- [x] Catat bahwa subset 834 kelas diberikan langsung oleh pihak pemilik dataset.
- [ ] Arsipkan izin penggunaan atau bukti penyerahan subset.
- [ ] Minta konfirmasi tertulis jika akan menyatakan alasan 266 kelas tidak dibagikan.
- [ ] Temukan metadata sesi/urutan atau nyatakan bahwa metadata tersebut tidak tersedia.
- [x] Pertahankan split 8/1/1 yang sama untuk seluruh model dan *seed*.
- [ ] Simpan hash file split dalam setiap artefak hasil.
- [ ] Bekukan satu model Ding utama dan status provenance implementasinya.
- [ ] Validasi provenance bobot setiap model *pretrained*.
- [ ] Jalankan Eksperimen 1: *scratch* 600 epoch maksimum × 3 *seed*.
- [ ] Pilih dua penerima KD menggunakan hasil validasi, bukan data uji.
- [ ] Jalankan Eksperimen 2: *pretrained* 200 epoch maksimum × 3 *seed*.
- [ ] Jalankan Eksperimen 3: KD 600 epoch maksimum × 3 *seed*.
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

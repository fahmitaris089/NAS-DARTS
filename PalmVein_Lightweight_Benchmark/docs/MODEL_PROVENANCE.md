# Model provenance and reconstruction audit

## Aturan umum adaptasi

Semua model scratch menerima tensor `[N, 3, 224, 224]` dan menghasilkan logits `[N, 834]`. Citra asli grayscale direplikasi menjadi tiga kanal di data loader; stem arsitektur eksternal tidak diubah. Classifier ImageNet 1.000 kelas diganti menjadi 834 kelas. Scratch selalu menggunakan inisialisasi baru. Bobot ImageNet hanya dimuat pada protokol `pretrained` dan tidak pernah dicampur ke tabel scratch.

## ProxylessNAS-Mobile

- sumber primer: MIT HAN Lab, `mit-han-lab/proxylessnas`;
- commit yang diaudit: `b23018c9c369d22931f7422b71ca6a7eaa354c46`;
- konfigurasi: `proxyless_mobile.config` dari repositori berkas resmi Han Cai;
- lisensi sumber: MIT;
- implementasi lokal: susunan 22 blok, termasuk dua `ZeroLayer`, dipindahkan ke `src/models/proxylessnas.py` dengan nama modul kompatibel terhadap state dict resmi;
- pralatih: URL `proxyless_mobile.pth` yang dirujuk oleh model zoo resmi; load menggunakan `strict=True` sebelum classifier diganti.

## FBNet-C

- sumber primer: Meta Mobile Vision, `facebookresearch/mobile-vision`;
- commit yang dipin: `7ed7e9177482140f58b5a56cc1acf54ecb4c1326`;
- konfigurasi: `mobile_cv/arch/fbnet_v2/fbnet_modeldef_cls_fbnet.py`, varian `fbnet_c`;
- catatan lisensi: repositori menyatakan CC BY-NC; periksa kesesuaian penggunaan sebelum distribusi komersial;
- scratch: rekonstruksi lokal atas urutan blok resmi agar benchmark tidak membutuhkan unduhan jaringan;
- pralatih: model dan bobot dibuat langsung oleh `mobile_cv.model_zoo.models.fbnet_v2.fbnet("fbnet_c", pretrained=True)`, lalu head diganti menjadi 834 kelas. Dependency dipin pada commit di `requirements.txt`.

## MnasNet-A1

- sumber primer: M. Tan et al., "MnasNet: Platform-Aware Neural Architecture Search for Mobile," Figure 7;
- status: implementasi lokal berdasarkan definisi arsitektur yang diterbitkan, bukan implementasi resmi penulis;
- konfigurasi: stem 32; depthwise-separable 16; MBConv `24/40/80/112/160/320`; kernel, repeat, expansion, dan SE ratio 0,25 mengikuti Figure 7; head 1.280 dan dropout 0,2;
- konfigurasi BatchNorm mengikuti kode resmi penulis: epsilon `1e-3` dan momentum TensorFlow `0,99`, yang diimplementasikan sebagai momentum PyTorch `0,01`;
- parameter: 3.887.038 untuk 1.000 kelas dan 3.674.392 setelah classifier diadaptasi menjadi 834 kelas;
- audit independen: urutan shape seluruh tensor diperiksa terhadap `timm==1.0.27` varian `semnasnet_100`, yang juga merujuk implementasi TensorFlow TPU penulis;
- protokol: scratch saja. Bobot `timm` tidak digunakan dan model tidak dimasukkan ke tabel pretrained.

Kecocokan parameter dan shape tensor mendukung kesetaraan struktur implementasi, tetapi tidak mengubah statusnya menjadi implementasi resmi dan tidak menjamin reproduksi akurasi ImageNet paper.

## MnasNet-B1 (torchvision `mnasnet1_0`)

- sumber API dan bobot: `pytorch/vision` tag `v0.21.0`, commit `7af698794eded568735f9519593603c1ec889eba`;
- implementasi lokal mempertahankan urutan shape seluruh tensor state dict yang identik dengan `torchvision.mnasnet1_0`;
- parameter: 4.383.312 untuk 1.000 kelas dan 4.170.666 untuk 834 kelas;
- bobot pralatih: `MNASNet1_0_Weights.IMAGENET1K_V1`, dimuat dengan `strict=True` sebelum classifier diganti;
- hasil yang sebelumnya disimpan dengan ID `mnasnet_a1` dipertahankan dan dilabel ulang sebagai `mnasnet_b1_torchvision`. Hash tensor model dicatat dalam manifest migrasi.

Model ini dipertahankan sebagai analisis transfer tambahan dan tidak disajikan sebagai hasil MnasNet-A1.

## DingBaseline, DingPW, dan DingPruned

Status ketiganya: **paper-constrained independent reconstruction**. Kode penulis tidak tersedia dalam material yang diaudit, sehingga model tidak boleh disebut implementasi resmi atau replikasi persis. Rekonstruksi mengikuti struktur enam blok, operasi PW, konfigurasi kanal model akhir, dan urutan pooling yang dapat dipertanggungjawabkan dari paper.

Struktur yang digunakan:

- DingBaseline memakai enam blok Conv3×3–BN–ReLU dengan kanal `[32, 32, 64, 64, 128, 128]`;
- DingPW mempertahankan tiga blok awal baseline dan mengganti tiga blok terakhir dengan `Conv1×1 → Conv3×3 → Conv1×1`;
- bottleneck DingPW pada blok 4–6 adalah `[32, 16, 64]`, dengan keluaran `[64, 128, 128]`;
- DingPruned mengikuti kanal Tabel II: keluaran `[22, 22, 44, 44, 89, 89]` dan bottleneck `[22, 11, 44]` pada tiga blok terakhir;
- MaxPool2×2 diterapkan setelah lima blok pertama, sedangkan blok keenam langsung menuju global average pooling dan classifier;
- tidak ada depthwise convolution pada ketiga rekonstruksi.

Paper melaporkan konfigurasi kanal akhir dan tahap Conv–BN fusion, tetapi tidak merinci posisi BN pada setiap konvolusi internal dalam blok PW. Rekonstruksi untuk pelatihan menempatkan satu BN dan ReLU setelah konvolusi terakhir pada setiap blok konseptual. Keputusan ini mengikuti keterangan paper bahwa setiap blok CONV mencakup convolution dan BN, tetapi tetap dicatat sebagai asumsi implementasi.

Jumlah parameter konfigurasi referensi satu kanal/500 kelas adalah 351.380, 165.268, dan 92.986 untuk Baseline, PW, dan Pruned. Angka tersebut konsisten dengan nilai paper yang dibulatkan menjadi 0,351M, 0,165M, dan 0,093M. Audit operasi Conv/Linear menghasilkan 238,50M, 202,02M, dan 98,61M operasi perkalian-akumulasi menurut konvensi penghitung lokal. Nilai terakhir dibagi dua menjadi 49,31M, sama dengan angka model hasil fusion pada paper. Hal ini menunjukkan perbedaan konvensi pelaporan kompleksitas, sehingga angka MAC paper dan benchmark lokal tidak dibandingkan tanpa menyebut alat serta definisi penghitungannya.

Paper mencantumkan MaxPool setelah blok keenam sekaligus AvgPool7×7 dari feature map 7×7. Kedua operasi tersebut tidak konsisten secara spasial. Implementasi menghilangkan MaxPool terakhir dan mempertahankan pooling 7×7 melalui adaptive global average pooling. Keputusan ini merupakan asumsi rekonstruksi yang dinyatakan terbuka.

Konfigurasi satu kanal/500 kelas disediakan untuk audit struktur paper. Benchmark terkontrol memakai citra grayscale yang direplikasi menjadi tiga kanal, classifier 834 kelas, serta normalisasi ImageNet yang sama dengan model lain. Jumlah parameter dan MMAC dicatat sebagai konsekuensi struktur, bukan digunakan untuk mengubah topologi agar menyerupai angka paper.

Implementasi lima blok berbasis depthwise–pointwise yang sebelumnya memakai ID `ding_pruned` telah dipindahkan ke `ding_pruned_legacy_parameter_matched_v1`. Hasil tersebut dipertahankan untuk audit, tetapi dikeluarkan dari ringkasan benchmark utama.

## PalmNet

Sumber primer: S. Luo dan X. Huang, “A Lightweight Neural Network for Palm Vein Recognition,” *Frontiers in Computing and Intelligent Systems*, 2022, doi: `10.54097/fcis.v2i3.5412`.

Status implementasi: **paper-constrained independent reconstruction**. Paper dan materi publik yang diaudit tidak menyediakan kode resmi atau konfigurasi lapisan lengkap. Oleh karena itu, model lokal tidak boleh disebut implementasi resmi, reproduksi eksak, atau bukti reproduksi angka akurasi paper.

Bagian yang dinyatakan oleh paper:

- urutan tahap ShuffleNetV2, MobileNetV3, dan MBConv;
- kode empat digit menyatakan jumlah blok ShuffleNetV2, jumlah blok MobileNetV3, jumlah MBConv, dan faktor ekspansi;
- blok ShuffleNetV2 menggunakan pemisahan kanal, depthwise convolution, concatenation, dan channel shuffle;
- blok MobileNetV3 dan MBConv menggunakan inverted bottleneck, depthwise convolution, SE, projection, serta residual ketika bentuk tensor sesuai;
- keluaran jaringan melewati convolution, Swish, global average pooling, dan classifier;
- `PalmNet-0.5x2413` dilaporkan memiliki sekitar 0,49M parameter dan 43,40M FLOPs, sedangkan `PalmNet-0.5x2411` sekitar 0,39M parameter dan 24,83M FLOPs.

Bagian yang direkonstruksi karena tidak ditentukan lengkap oleh paper:

- kanal tahap mengikuti jadwal standar ShuffleNetV2: `[24, 48, 96, 192, 1024]` untuk lebar 0,5, `[24, 116, 232, 464, 1024]` untuk lebar 1,0, dan `[24, 244, 488, 976, 2048]` untuk lebar 2,0;
- blok pertama setiap tahap menggunakan stride 2 dan blok berikutnya stride 1;
- stem menggunakan Conv3x3–BN–ReLU stride 2 dan MaxPool3x3 stride 2;
- SE ratio ditetapkan 0,25;
- DropPath dibuat konfigurabel dan dinonaktifkan pada benchmark;
- MobileNetV3 memakai ReLU, sedangkan MBConv dan head memakai SiLU/Swish;
- input referensi memakai satu kanal dan classifier 200 kelas hanya untuk audit struktur.

Adaptasi benchmark memakai input tiga kanal, resolusi 224x224, classifier 834 kelas, dan normalisasi ImageNet yang sama dengan model lain. Nilai parameter dan MMAC lokal dibandingkan dengan tabel paper hanya sebagai diagnostik. Selisih tidak digunakan untuk mengubah kanal secara arbitrer karena konfigurasi lapisan dan resolusi input paper tidak diterbitkan secara lengkap.

Model utama adalah `palmnet_05x_2413`; `palmnet_05x_2411` menjadi pembanding efisiensi. Seluruh kode tabel paper dapat dibangun melalui `src/models/palmnet.py`, tetapi varian selain kedua model utama hanya dijalankan jika dipilih secara eksplisit. PalmNet tidak mempunyai bobot pralatih resmi yang diaudit dan hanya mendukung protokol scratch.

`src/models/chen.py` dan konfigurasi Chen lama dipertahankan sebagai artefak audit, tetapi model tersebut tidak terdaftar di factory dan tidak masuk matriks benchmark utama. Efek KD tetap dianalisis melalui P-DARTS tanpa KD dan P-DARTS dengan KD.

## P-DARTS L0.05 C12 cells10

- sumber lokal genotype: `../nas_results/retrain_hwNAS_L0.05_C12_cells10_stemds8_834cls/config.json`;
- commit repositori sumber: `f940a8ed04693dea3f0a887b3ef0fe3140ef482b`;
- SHA-256 sumber saat ekstraksi: `975dfa6ac8cc0cb833d9191f385c3d5ffe32873a4b761ac14c95b20fc9c01419`;
- konfigurasi benchmark: `C_init=12`, `num_cells=10`, `stem_downsample=8`, reduction cell pada indeks 3 dan 7;
- operator yang disertakan: `rep_conv_3x3`, `dil_conv_3x3`, dan `skip_connect`;
- classifier: 834 kelas;
- training benchmark: bobot baru; hasil checkpoint tesis lama tidak digunakan.

Konfigurasi sumber lama memuat beberapa nilai bersarang yang bertentangan dengan field run aktual. Benchmark hanya mengambil genotype dan field arsitektur eksplisit pada level atas; hiperparameter training diganti sepenuhnya oleh protokol benchmark terkontrol.

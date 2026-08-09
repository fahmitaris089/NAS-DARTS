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

- konfigurasi acuan: MnasNet-A1 (stem 32, depthwise-separable 16, lalu blok 24/40/80/96/192/320 dan head 1.280);
- sumber bobot/API yang diaudit: `pytorch/vision` tag `v0.21.0`, commit `7af698794eded568735f9519593603c1ec889eba`;
- implementasi lokal: `src/models/mnasnet.py`;
- audit torchvision: implementasi lokal dan `torchvision.mnasnet1_0` wajib memiliki urutan shape seluruh tensor state dict yang identik. Pada konfigurasi 1.000 kelas keduanya memiliki 4.383.312 parameter. Bobot tidak dimuat jika audit shape gagal;
- pralatih: `MNASNet1_0_Weights.IMAGENET1K_V1`, kemudian classifier diganti menjadi 834 kelas.

Audit ini mengatasi risiko menggunakan `torchvision.mnasnet1_0` hanya berdasarkan kemiripan nama. Kesetaraan yang diperiksa adalah topologi tensor/parameter; kesetaraan numerik diverifikasi saat pemetaan state dict dengan `strict=True`.

## DingBaseline, DingPW, dan DingPruned

Status ketiganya: **paper-constrained independent reconstruction**. Kode penulis tidak tersedia dalam material yang diaudit. Karena publikasi hanya cukup mendukung envelope parameter/kompresi, topologi lokal tidak boleh disebut implementasi resmi atau replikasi persis.

Asumsi eksplisit:

- DingBaseline menggunakan lima blok convolution–BN–ReLU dengan kanal `[16, 32, 64, 112, 176]`;
- DingPW mengganti blok setelah stem dengan depthwise–pointwise dan kanal `[16, 32, 64, 128, 239]`;
- DingPruned menggunakan depthwise–pointwise dengan kanal `[12, 24, 48, 72, 144]`;
- semua versi memakai global average pooling dan classifier linear;
- konfigurasi referensi 1 kanal/500 kelas digunakan hanya untuk audit parameter; benchmark memakai 3 kanal/834 kelas.

Hasil audit referensi yang diharapkan dari kode lokal:

| Model | Parameter lokal | Target paper-level | Deviasi |
|---|---:|---:|---:|
| DingBaseline | 354.404 | 351.000 | +0,97% |
| DingPW | 165.086 | 165.000 | +0,05% |
| DingPruned | 90.188 | 93.000 | −3,02% |

Kedekatan parameter tidak membuktikan kesetaraan topologi atau performa. Perbandingan hasil harus menyebut status rekonstruksi ini sebagai keterbatasan validitas konstruk.

## P-DARTS L0.05 C12 cells10

- sumber lokal genotype: `../nas_results/retrain_hwNAS_L0.05_C12_cells10_stemds8_834cls/config.json`;
- commit repositori sumber: `f940a8ed04693dea3f0a887b3ef0fe3140ef482b`;
- SHA-256 sumber saat ekstraksi: `975dfa6ac8cc0cb833d9191f385c3d5ffe32873a4b761ac14c95b20fc9c01419`;
- konfigurasi benchmark: `C_init=12`, `num_cells=10`, `stem_downsample=8`, reduction cell pada indeks 3 dan 7;
- operator yang disertakan: `rep_conv_3x3`, `dil_conv_3x3`, dan `skip_connect`;
- classifier: 834 kelas;
- training benchmark: bobot baru; hasil checkpoint tesis lama tidak digunakan.

Konfigurasi sumber lama memuat beberapa nilai bersarang yang bertentangan dengan field run aktual. Benchmark hanya mengambil genotype dan field arsitektur eksplisit pada level atas; hiperparameter training diganti sepenuhnya oleh protokol benchmark terkontrol.

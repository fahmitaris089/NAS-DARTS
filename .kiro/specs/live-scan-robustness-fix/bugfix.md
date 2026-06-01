# Bugfix Requirements Document — Live Scan Robustness Fix

## Introduction

Sistem palm vein recognition berbasis NAS-DARTS yang sudah di-retrain pada
dataset milik user (kelas `835` = tangan kiri, kelas `836` = tangan kanan)
menunjukkan defect pada deployment / live scan. Defect ini tidak muncul saat
evaluasi dengan training/test split internal karena distribusi training dan
test sama (semua diambil pada jarak statik 27 cm dengan satu konfigurasi
illumination), tetapi muncul saat user melakukan live scan dengan kondisi
yang sedikit bervariasi.

Dua simptom utama yang teramati pada live scan:

1. **Distance sensitivity.** Saat jarak tangan ke kamera tidak persis 27 cm
   (jarak yang dipakai mengambil seluruh training data), sistem memberikan
   reject (gagal lulus quality filter / confidence + margin di bawah
   threshold) atau memprediksi class yang salah, sehingga pengguna sah
   tidak dapat di-recognize.
2. **Class confusion antara tangan kiri dan kanan.** Pada beberapa scan
   tangan kanan (ground truth = `836`), sistem accept dengan
   `predicted_subject = 835`, dan sebaliknya untuk tangan kiri. Padahal kedua
   kelas tersebut adalah subject yang berbeda secara identitas (left vs
   right hand), dan pertukaran ini melanggar invarian fundamental sistem
   biometrik.

Akar sumber defect (akan diformalkan di `design.md`) berkaitan dengan domain
gap antara distribusi training (statik, 2 kelas, ~10 sampel/kelas, tanpa
variasi jarak/rotasi/illumination) dan distribusi deployment (variable
distance, kemungkinan rotasi tangan, kondisi illumination yang sedikit
berbeda walau exposure dijaga). Bugfix ini mendefinisikan perilaku yang
harus diperbaiki tanpa meregresikan perilaku capture/preprocess yang sudah
benar pada kondisi nominal.

## Bug Analysis

### Current Behavior (Defect)

Berikut perilaku yang saat ini salah saat menggunakan model NAS-DARTS hasil
retrain pada live scan:

1.1 WHEN live scan dilakukan pada jarak tangan-ke-kamera yang berbeda dari
27 cm (di luar toleransi sempit yang implisit dipelajari model dari training
data) THEN the system menolak input meskipun pengguna adalah subject
terdaftar (false reject) atau memberikan `predicted_subject` yang tidak
sesuai dengan ground truth identity.

1.2 WHEN live scan dilakukan untuk tangan kanan user (ground truth identity
= subject `836`) pada kondisi yang lulus quality filter THEN the system
kadang accept dengan `predicted_subject = 835` (kelas tangan kiri),
melakukan misidentification antar tangan dari subject yang sama.

1.3 WHEN live scan dilakukan untuk tangan kiri user (ground truth identity
= subject `835`) pada kondisi yang lulus quality filter THEN the system
kadang accept dengan `predicted_subject = 836` (kelas tangan kanan),
melakukan misidentification dengan arah berlawanan dari klausul 1.2.

1.4 WHEN input live scan berada pada jarak yang merupakan domain shift
(distribusi deployment ≠ distribusi training) THEN the system tidak membuat
distinction yang explicit antara "input out-of-distribution" dan "input
in-distribution tetapi bukan subject terdaftar"; reject yang terjadi tidak
membawa reason yang dapat dibedakan, sehingga user tidak tahu apakah harus
memperbaiki posisi tangan atau memang ditolak sebagai non-enrolled.

1.5 WHEN model dievaluasi pada training/test split internal saja THEN the
system menunjukkan akurasi tinggi yang menyembunyikan bug 1.1–1.4, karena
test split memiliki distribusi yang sama dengan train split (statik 27 cm,
kondisi sama), sehingga bug hanya manifest di deployment.

### Expected Behavior (Correct)

Perilaku yang seharusnya pada live scan setelah fix diterapkan:

2.1 WHEN live scan dilakukan pada jarak dalam range operasional yang
tervalidasi (range eksplisit yang mencakup 27 cm dan toleransi yang akan
ditentukan di design phase, mis. 22–32 cm) untuk subject terdaftar THEN
the system SHALL accept input dan memberikan `predicted_subject` yang sama
dengan ground truth identity, dengan accuracy yang memenuhi target yang
disepakati di design phase.

2.2 WHEN live scan dilakukan untuk tangan kanan (ground truth = `836`)
dalam range jarak operasional yang tervalidasi THEN the system SHALL
memprediksi `predicted_subject = 836` dan bukan `835`, sehingga left-vs-right
hand discrimination konsisten dengan ground truth.

2.3 WHEN live scan dilakukan untuk tangan kiri (ground truth = `835`)
dalam range jarak operasional yang tervalidasi THEN the system SHALL
memprediksi `predicted_subject = 835` dan bukan `836`.

2.4 WHEN input live scan berasal dari kondisi yang berada di luar range
operasional yang tervalidasi (out-of-distribution distance, illumination
ekstrem, atau pose tangan di luar yang dilatihkan) THEN the system SHALL
melakukan reject dengan reason yang explicit dan dapat dibedakan dari
reject akibat low-confidence/low-margin pada input in-distribution
(misalnya reason berupa `out_of_distribution` atau `domain_shift_detected`),
sehingga sistem fail-safe dan tidak melakukan misclassification silent.

2.5 WHEN evaluasi end-to-end dilakukan THEN the system SHALL dievaluasi
pada distribusi yang merepresentasikan kondisi deployment (multi-distance,
multi-pose, multi-illumination dalam range yang ditetapkan), sehingga
metrics yang dilaporkan mencerminkan robustness sebenarnya, bukan akurasi
artifisial dari evaluasi statik.

2.6 WHEN model membuat decision di kondisi yang mengandung domain shift
yang dapat dideteksi (mis. jarak di luar range training/enrollment, ROI
yang ukurannya menyimpang dari distribusi normal) THEN the system SHALL
fail-safe ke reject dengan reason explicit, dan SHALL TIDAK accept dengan
predicted_subject yang dapat menyebabkan cross-hand confusion seperti
1.2/1.3.

### Unchanged Behavior (Regression Prevention)

Perilaku berikut harus dipertahankan setelah fix; klausul ini melindungi
kontrak yang sudah benar pada pipeline saat ini:

3.1 WHEN live scan dilakukan pada jarak 27 cm (kondisi nominal yang
dipakai mengambil training data) untuk subject `835` dengan tangan kiri
yang berkualitas (lulus quality filter, exposure dan illumination sesuai
profil dataset_v3) THEN the system SHALL CONTINUE TO accept dengan
`predicted_subject = 835`.

3.2 WHEN live scan dilakukan pada jarak 27 cm (kondisi nominal yang
dipakai mengambil training data) untuk subject `836` dengan tangan kanan
yang berkualitas (lulus quality filter, exposure dan illumination sesuai
profil dataset_v3) THEN the system SHALL CONTINUE TO accept dengan
`predicted_subject = 836`.

3.3 WHEN konfigurasi illumination dan exposure tuning (NIR 850 nm dengan
satu lembar tisu sebagai diffuser, `exposure-us 8000`, `gain 1.1`,
`awbgains 1.0,1.0`, `brightness -0.04`, `contrast 1.3`, `saturation 0`)
dipertahankan THEN the system SHALL CONTINUE TO menghasilkan capture
grayscale yang lulus quality filter `dataset_v3` (laplacian-var min 60)
pada jarak 27 cm.

3.4 WHEN pipeline preprocessing `dataset_v3` (adaptive ROI dengan
`adaptive_roi_scale = 0.95`, `palm_core_width_ratio = 0.45`, CLAHE
`clip_limit = 2.4`, `tile_grid = (8,8)`, final size 224×224)
dijalankan pada training images yang sudah ada di
`dataset/835/*.bmp` dan `dataset/836/*.bmp` THEN the system SHALL
CONTINUE TO menghasilkan ROI dan final image dengan dimensi dan
karakteristik intensity yang sama dengan sebelum fix (deterministic
output untuk input yang sama), sehingga template store dan model
weights yang sudah di-train tidak invalidated tanpa retrain explicit.

3.5 WHEN ONNX inference dijalankan pada image yang lolos preprocessing
(input shape `[1, 3, 224, 224]`, normalisasi ImageNet) di kondisi
nominal THEN the system SHALL CONTINUE TO mengeluarkan `logits` dan
`embedding` outputs yang shape dan range distribusinya konsisten
dengan kontrak ONNX metadata yang ada saat ini.

3.6 WHEN decision rule diterapkan pada input yang in-distribution
(non-buggy: jarak ≈ 27 cm, illumination nominal, hand-side yang
benar) THEN the system SHALL CONTINUE TO menerapkan kombinasi
quality filter + confidence threshold + margin threshold + consensus
window seperti yang sudah berjalan; fix tidak boleh melonggarkan
gating untuk input in-distribution sehingga false-accept rate naik.

3.7 WHEN enrollment template untuk subject `835` dan `836` yang
sudah ada (live_enroll_left/, live_enroll_right/, hasil
`enroll_templates_onnx.py`) dievaluasi pada query yang berasal dari
distribusi training yang sama (jarak 27 cm) THEN the system SHALL
CONTINUE TO menghasilkan cosine similarity yang berada di range
acceptance yang sama dengan baseline (tidak ada degradasi pada
in-distribution recognition akibat perubahan template/model).

3.8 WHEN konfigurasi capture script `capture_on_hand_detect.py`
dijalankan dengan parameter capture default (`size 1920x1080`,
`fps 30`, profile `dataset_v3`) THEN the system SHALL CONTINUE TO
mendeteksi tangan, melakukan stable-frame burst, memilih best frame,
menyimpan raw + processed output, dan menulis metadata JSON dengan
struktur yang sama dengan saat ini (kontrak file output tidak break).

### Bug Condition Formalization (Specification Level)

Definisi formal dari bug yang akan dipakai sebagai dasar property-based
verification di phase design/tasks. Definisi ini bersifat spesifikasi,
bukan implementasi.

```pascal
// X mewakili satu live-scan event dengan atribut:
//   X.image    : raw grayscale capture
//   X.distance : jarak tangan-ke-kamera saat capture (cm)
//   X.side     : "left" atau "right" (ground truth hand-side)
//   X.subject  : ground truth subject id ("835" atau "836")
//   X.in_quality_band : boolean, true jika capture lulus quality filter

FUNCTION isBugCondition(X)
  INPUT: X of type LiveScanEvent
  OUTPUT: boolean

  // Domain shift jarak: di luar range operasional D_op yang akan
  // ditetapkan di design phase (D_op mencakup 27 cm + toleransi).
  let DISTANCE_SHIFT  := X.distance NOT IN D_op

  // Class-confusion left/right: kondisi di mana model historisnya
  // dapat menukar 835 dengan 836 walaupun X.in_quality_band = true.
  // Kondisi ini adalah seluruh ruang input yang berisiko cross-hand.
  let CROSS_HAND_RISK := (X.subject IN {"835","836"})
                        AND (X.in_quality_band = true)

  RETURN DISTANCE_SHIFT OR CROSS_HAND_RISK
END FUNCTION
```

**Property — Fix Checking** (untuk input yang trigger bug, perilaku
fixed function `F'` harus benar):

```pascal
// F'(X) = (decision, predicted_subject, reason)
// Untuk distance shift: F' boleh accept jika X.distance ada dalam
// range tervalidasi yang lebih luas dari training, atau reject
// dengan reason explicit "out_of_distribution".
FOR ALL X WHERE isBugCondition(X) DO
  result ← F'(X)

  IF X.distance IN D_validated THEN
    // accept with correct subject
    ASSERT result.decision = "accepted"
       AND result.predicted_subject = X.subject
  ELSE IF X.distance NOT IN D_validated THEN
    // out-of-distribution: must reject with explicit reason
    ASSERT result.decision = "rejected"
       AND result.reason CONTAINS "out_of_distribution"
  END IF

  // Cross-hand invariant — wajib true untuk semua X yang lulus
  // quality filter, terlepas dari distance:
  IF X.in_quality_band AND result.decision = "accepted" THEN
    ASSERT result.predicted_subject = X.subject
    // i.e. 835 -> 835 dan 836 -> 836, tidak boleh swap.
  END IF
END FOR
```

**Property — Preservation Checking** (untuk input non-buggy, fixed
function harus identik dengan original function `F`):

```pascal
// Non-buggy: jarak nominal training (≈ 27 cm) DAN tidak berisiko
// cross-hand (atau ground truth side cocok dengan prediksi yang
// sudah benar pada baseline).
FOR ALL X WHERE NOT isBugCondition(X) DO
  ASSERT F'(X) = F(X)
END FOR
```

**Definisi pendukung** (akan di-instansiasi konkret di `design.md`):

| Simbol            | Arti                                                          |
|-------------------|---------------------------------------------------------------|
| `F`               | Pipeline live scan saat ini (preprocess + ONNX + decision).   |
| `F'`              | Pipeline live scan setelah fix.                               |
| `D_op`            | Range jarak operasional sempit di sekitar 27 cm (≈ training). |
| `D_validated`     | Range jarak yang model fix-nya dilatih/dikalibrasi handle.    |
| `in_quality_band` | Lolos quality filter `dataset_v3` (laplacian-var ≥ 60).       |

Range numerik konkret untuk `D_op` dan `D_validated`, target accuracy,
dan threshold acceptance akan ditetapkan di phase Design beserta
strategi mitigasi (augmentation, multi-distance enrollment, OOD
detection, dst.). Dokumen ini hanya menetapkan kontrak perilaku yang
harus dipenuhi.

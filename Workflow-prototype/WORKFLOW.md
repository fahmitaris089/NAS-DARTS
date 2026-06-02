# Palm Vein Recognition — Workflow Prototype

Dokumentasi end-to-end dari akuisisi dataset hingga menjalankan prototype recognition
di Raspberry Pi menggunakan model NAS-DARTS.

---

## Prasyarat

### Server (GPU — vast.ai atau sejenisnya)
```bash
pip install -r requirements.txt
```

### Raspberry Pi 5 (untuk akuisisi + recognition)
```bash
sudo apt install -y python3-opencv python3-picamera2
pip install onnxruntime numpy opencv-python-headless
```

---

## Struktur Folder

```
Workflow-prototype/
├── WORKFLOW.md
├── nas_results/                        # Output retrain
│   └── retrain_<nama_run>/
│       ├── best_model.pth              ← Checkpoint PyTorch terbaik
│       ├── config.json                 ← Konfigurasi training
│       ├── model_benchmark.onnx        ← Dibuat di Step 6
│       └── model_benchmark_metadata.json
├── dataset_multi_distance/             # Dataset per subject & jarak
│   ├── 835/
│   │   ├── 22cm/  25cm/  ...          # Raw PNG hasil kamera (Step 1)
│   │   ├── final_raw/                 # 10 gambar terbaik per jarak (Step 2)
│   │   │   ├── 22cm/ ... 32cm/
│   │   └── final/                     # BMP 224×224 hasil preprocessing (Step 3)
│   │       ├── 22cm/ ... 32cm/
│   └── 836/
│       └── ...
│
├── capture_on_hand_detect.py          # [Pi]     Step 1: Akuisisi dataset
├── select_best_raw_images.py          # [Server] Step 2: Pilih 10 terbaik
├── preprocess_multi_distance_dataset.py  # [Server] Step 3: Preprocessing
├── build_multi_distance_split.py      # [Server] Step 4: Build split
├── retrain_nas.py                     # [Server] Step 5: Retrain NAS
├── retrain.py                         #           └─ core training loop
├── export_nas_onnx.py                 # [Server] Step 6: Export ke ONNX
├── prototype_nas_recognition_onnx.py  # [Pi]     Step 7: Running recognition
└── (library: model_eval.py, genotypes.py, palm_preprocessing.py, ...)
```

---

## Step 1: Akuisisi Dataset (Raspberry Pi)

Jalankan di **Raspberry Pi**. Ulangi untuk setiap subject dan jarak.

```bash
SUBJECT=835          # ID subject (835 atau 836)
DIST=22              # Jarak dalam cm (22 / 25 / 27 / 30 / 32)

python3 capture_on_hand_detect.py \
  --size 1920x1080 \
  --fps 30 \
  --exposure-us 8000 \
  --gain 1.1 \
  --awbgains 1.0,1.0 \
  --brightness -0.04 \
  --contrast 1.3 \
  --saturation 0 \
  --out-dir dataset_multi_distance/${SUBJECT}/${DIST}cm \
  --stable-frames 12 \
  --burst-frames 10 \
  --preprocess \
  --preprocess-profile dataset_v3 \
  --quality-filter \
  --quality-min-laplacian-var 60 \
  --save-rejected
```

### Setting kamera per jarak

| Jarak    | `--exposure-us` | `--gain` | `--contrast` |
|----------|-----------------|----------|--------------|
| 22–25 cm | `6000`          | `1.0`    | `1.5`        |
| 27–30 cm | `8000`          | `1.1`    | `1.3`        |
| 32 cm+   | `9000`          | `1.2`    | `1.3`        |

### Penjelasan argumen

| Argumen | Keterangan |
|---------|------------|
| `--size 1920x1080` | Resolusi full HD |
| `--fps 30` | Frame rate kamera |
| `--exposure-us` | Waktu eksposur (µs) — sesuaikan per jarak (tabel di atas) |
| `--gain` | Analog gain sensor — naikkan jika gambar terlalu gelap |
| `--awbgains 1.0,1.0` | White balance manual (red,blue) — tetap 1.0,1.0 untuk NIR |
| `--brightness -0.04` | Sedikit gelap agar kontras pembuluh darah terlihat |
| `--saturation 0` | Grayscale efektif untuk kamera NIR |
| `--out-dir` | Folder output — ubah sesuai subject dan jarak |
| `--stable-frames 12` | Jumlah frame stabil sebelum trigger burst |
| `--burst-frames 10` | Jumlah frame yang diambil per deteksi |
| `--preprocess` | Aktifkan preprocessing inline |
| `--preprocess-profile dataset_v3` | Profil preprocessing konsisten dengan training |
| `--quality-filter` | Filter gambar berkualitas rendah |
| `--quality-min-laplacian-var 60` | Minimum sharpness (Laplacian variance) |
| `--save-rejected` | Simpan frame rejected untuk debugging |

### Output struktur
```
dataset_multi_distance/835/
├── 22cm/
│   ├── palm_20260601_215629_835861.png   ← raw PNG accepted
│   └── rejected/                         ← frame yang dibuang
├── 25cm/ ... 32cm/
```

**Target:** minimal 10–15 gambar accepted per jarak per subject.

---

## Step 2: Pilih 10 Raw Image Terbaik (Server/Laptop)

Memilih 10 gambar terbaik per jarak berdasarkan **Laplacian variance** (sharpness proxy).

```bash
# Subject 835
python3 select_best_raw_images.py \
    --dataset-root dataset_multi_distance/835 \
    --output-dir   dataset_multi_distance/835/final_raw \
    --samples-per-distance 10

# Subject 836
python3 select_best_raw_images.py \
    --dataset-root dataset_multi_distance/836 \
    --output-dir   dataset_multi_distance/836/final_raw \
    --samples-per-distance 10
```

### Penjelasan argumen

| Argumen | Keterangan |
|---------|------------|
| `--dataset-root` | Root folder subject (berisi subfolder 22cm, 25cm, dst.) |
| `--output-dir` | Folder output gambar terpilih |
| `--samples-per-distance` | Jumlah gambar terbaik per jarak (default: 10) |
| `--dry-run` | Preview selection tanpa copy file |

### Output struktur
```
dataset_multi_distance/835/final_raw/
├── 22cm/
│   ├── palm_20260601_215629_835861.png   ← 10 gambar terbaik
│   └── selection_report.json
├── 25cm/ ... 32cm/
```

---

## Step 3: Preprocessing Final (Server)

Mengkonversi raw PNG → preprocessed BMP 224×224 siap training.  
Pipeline: **Adaptive ROI → CLAHE → Min-max normalization → Resize 224×224**

```bash
# Subject 835
python3 preprocess_multi_distance_dataset.py \
    --input-root  dataset_multi_distance/835/final_raw \
    --output-root dataset_multi_distance/835/final \
    --subject-id  835

# Subject 836
python3 preprocess_multi_distance_dataset.py \
    --input-root  dataset_multi_distance/836/final_raw \
    --output-root dataset_multi_distance/836/final \
    --subject-id  836
```

### Penjelasan argumen

| Argumen | Keterangan |
|---------|------------|
| `--input-root` | Folder berisi subfolder jarak dengan raw PNG |
| `--output-root` | Folder output BMP yang sudah diproses |
| `--subject-id` | ID subject (835 atau 836) |

### Output struktur
```
dataset_multi_distance/835/final/
├── 22cm/
│   ├── palm_20260601_215629_835861.bmp              ← 224×224 grayscale
│   └── palm_20260601_215629_835861_preprocess.json  ← metadata preprocessing
├── 25cm/ ... 32cm/
```

---

## Step 4: Split Dataset (Server)

Membuat file JSON split train/val/test yang balanced antar subject dan jarak.

```bash
python3 build_multi_distance_split.py \
    --dataset-root  dataset_multi_distance \
    --output-file   dataset_multi_distance/split_info.json \
    --subjects      835 836 \
    --source-folder final \
    --train-ratio   0.6 \
    --val-ratio     0.2 \
    --test-ratio    0.2 \
    --seed          42
```

### Penjelasan argumen

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--dataset-root` | — | Root folder dataset (berisi subfolder per subject) |
| `--output-file` | — | Path output file JSON split |
| `--subjects` | `835 836` | Daftar subject ID (pisahkan dengan spasi) |
| `--source-folder` | `final` | Subfolder sumber dalam tiap subject |
| `--train-ratio` | `0.6` | Proporsi data training |
| `--val-ratio` | `0.2` | Proporsi data validasi |
| `--test-ratio` | `0.2` | Proporsi data test |
| `--seed` | `42` | Random seed untuk reproducibility |

### Struktur split_info.json yang dihasilkan
```json
{
  "dataset_root": "dataset_multi_distance",
  "source_folder": "final",
  "subjects": ["835", "836"],
  "label_map": {"835": 0, "836": 1},
  "splits": {
    "train": ["835/final/22cm/palm_....bmp", "..."],
    "val":   ["..."],
    "test":  ["..."]
  },
  "metadata": {
    "train": [{"path": "...", "subject_id": "835", "distance_cm": "22cm"}, "..."]
  }
}
```

---

## Step 5: Retrain NAS (Server GPU)

Script ini mempersiapkan data, membuat symlink, lalu memanggil `retrain.py`.

```bash
python3 retrain_nas.py \
    --split-file          dataset_multi_distance/split_info.json \
    --genotype            nas_results/search/genotype_final.json \
    --output-dir          nas_results/retrain_<nama_run> \
    --augmentation-policy v2_multi_distance \
    --epochs              300
```

### Argumen utama

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--split-file` | `dataset_multi_distance/split_info.json` | File split dari Step 4 |
| `--genotype` | `nas_results/search/genotype_final.json` | Arsitektur hasil NAS search |
| `--output-dir` | `nas_results/retrain_<run>` | Folder output training |
| `--augmentation-policy` | `v2_multi_distance` | `v1_legacy` (ada flip) atau `v2_multi_distance` (tanpa flip, lebih agresif) |
| `--epochs` | dari run6 config | Jumlah epoch training |
| `--C_init` | auto-tuned | Channel awal arsitektur (default: 16) |
| `--num_cells` | dari run6 config | Jumlah cells (default: 8) |
| `--batch_size` | dari run6 config | Batch size |
| `--lr` | dari run6 config | Learning rate |
| `--hand-pair-margin-loss` | — | Aktifkan margin loss untuk cross-hand discrimination |
| `--prepare-only` | — | Hanya siapkan data tanpa launch training |

### Preview tanpa training
```bash
python3 retrain_nas.py \
    --split-file  dataset_multi_distance/split_info.json \
    --output-dir  nas_results/retrain_test \
    --prepare-only
```

### Output struktur setelah training
```
nas_results/retrain_<nama_run>/
├── best_model.pth           ← Model dengan val accuracy terbaik
├── last_model.pth           ← Model epoch terakhir
├── config.json              ← Konfigurasi training
├── training_log.csv         ← Loss & accuracy per epoch
├── test_results.json        ← Hasil evaluasi test set
├── classification_report.txt
└── data_symlinks/           ← Symlink ke dataset (untuk training)
```

---

## Step 6: Export ke ONNX (Server)

Export checkpoint PyTorch hasil retrain → ONNX untuk inference di Pi tanpa PyTorch.

> **Prasyarat:** `nas_results/retrain_<nama_run>/` sudah berisi `best_model.pth`
> dan `config.json` dari Step 5.

```bash
python3 export_nas_onnx.py \
    --model-dir         nas_results/retrain_<nama_run> \
    --subjects          835 836 \
    --include-embeddings
```

> **Catatan:** `--subjects` wajib diisi jika menjalankan di luar server tempat training,
> karena `split_path` di `config.json` adalah absolute path server.

### Penjelasan argumen

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--model-dir` | — | Folder hasil retrain (berisi `best_model.pth` dan `config.json`) |
| `--subjects` | dari split file | Subject ID — wajib jika path config server tidak bisa diakses |
| `--include-embeddings` | ✓ default on | Sertakan output embedding 256-dim bersama logits |
| `--logits-only` | — | Hanya export logits (tanpa embedding) |
| `--opset` | `13` | ONNX opset version |
| `--output-path` | `<model-dir>/model_benchmark.onnx` | Path ONNX output |

### Output
```
nas_results/retrain_<nama_run>/
├── model_benchmark.onnx            ← ~1.2 MB, siap untuk deployment Pi
└── model_benchmark_metadata.json   ← Metadata: subjects, output names, size
```

### Transfer ke Raspberry Pi
```bash
scp nas_results/retrain_<nama_run>/model_benchmark.onnx \
    nas_results/retrain_<nama_run>/model_benchmark_metadata.json \
    pi@<ip_pi>:~/palm-vein/retrain_<nama_run>/
```

---

## Step 7: Running Prototype Recognition (Raspberry Pi)

Pastikan `model_benchmark.onnx` dan `model_benchmark_metadata.json` sudah ada
di folder `retrain_<nama_run>/` di Pi.

```bash
python3 prototype_nas_recognition_onnx.py \
    --model-dir        retrain_<nama_run> \
    --out-dir          recognition_results \
    --size             1920x1080 \
    --fps              30 \
    --exposure-us      8000 \
    --gain             1.1 \
    --awbgains         1.0,1.0 \
    --brightness       -0.04 \
    --contrast         1.3 \
    --saturation       0 \
    --stable-frames    12 \
    --burst-frames     10 \
    --no-quality-filter \
    --save-rejected \
    --reject-threshold                 0.72 \
    --reject-margin                    0.20 \
    --consensus-window                 3 \
    --consensus-min-agree              2 \
    --consensus-min-average-confidence 0.72 \
    --consensus-min-average-margin     0.20
```

### Penjelasan argumen

| Argumen | Nilai | Keterangan |
|---------|-------|------------|
| `--model-dir` | `retrain_<run>` | Folder berisi `model_benchmark.onnx` dan metadata |
| `--out-dir` | `recognition_results` | Folder output hasil recognition |
| `--decision-mode` | `logits` (default) | `logits` = classification; `verification` = one-vs-one similarity |
| `--reject-threshold` | `0.72` | Minimum confidence untuk accept |
| `--reject-margin` | `0.20` | Minimum selisih confidence antar kelas |
| `--no-quality-filter` | — | Nonaktifkan quality filter (direkomendasikan untuk live Pi camera) |
| `--quality-filter` | — | Aktifkan quality filter (lebih cocok untuk dataset bersih) |
| `--consensus-window` | `3` | Jumlah frame dalam window consensus |
| `--consensus-min-agree` | `2` | Minimal frame yang harus sepakat dalam window |
| `--consensus-min-average-confidence` | `0.72` | Rata-rata confidence minimum dalam window |
| `--save-rejected` | — | Simpan frame rejected untuk debugging |
| `--preview` | — | Tampilkan preview kamera (butuh layar/HDMI) |

### Tabel threshold rekomendasi

| Skenario | `--reject-threshold` | `--reject-margin` | Keterangan |
|----------|---------------------|------------------|------------|
| Ketat (keamanan tinggi) | `0.85–0.90` | `0.30` | FAR sangat kecil, lebih banyak reject |
| Normal | `0.80` | `0.25` | Balance antara FAR dan FRR |
| Relaxed (kalibrasi awal) | `0.72` | `0.20` | Cocok untuk tuning awal di Pi |

### Output struktur
```
recognition_results/
├── accepted/
│   ├── images/     ← Frame preprocessed yang diterima
│   └── metadata/   ← JSON detail: subject, confidence, threshold, dsb.
└── rejected/
    ├── images/
    └── metadata/
```

Hentikan dengan **Ctrl+C**.

---

## Troubleshooting

| Gejala | Penyebab | Solusi |
|--------|----------|--------|
| `REJECTED: low_confidence` | Confidence < threshold | Turunkan `--reject-threshold` ke 0.72, atau perbaiki posisi tangan |
| `REJECTED: quality_filter` | Quality filter terlalu ketat untuk live camera | Gunakan `--no-quality-filter` |
| `REJECTED: consensus_not_ready` | Model bergantian prediksi antar subject dalam burst | Gunakan `--consensus-window 3 --consensus-min-agree 2`, pastikan tangan diam |
| `FileNotFoundError: model_benchmark.onnx` | Model belum di-export atau belum ditransfer ke Pi | Jalankan ulang Step 6, lalu transfer ke Pi |
| Confidence rendah di jarak tertentu | Jarak tangan berbeda dari kondisi saat dataset dibuat | Sesuaikan `--exposure-us`, `--gain`, `--contrast` (lihat tabel Step 1) |

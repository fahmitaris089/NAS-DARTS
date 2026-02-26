# NAS Search Run #1 — Architecture Collapse Analysis

**Tanggal:** 25 Februari 2026  
**Durasi:** 123.3 menit  
**Status:** COLLAPSED — 100% max_pool_3x3

---

## 1. Ringkasan Hasil

| Parameter             | Nilai                                  |
| --------------------- | -------------------------------------- |
| Tanggal mulai         | 2026-02-25 08:43:20                    |
| Durasi total          | 123.3 menit                            |
| C_search              | **8** (terlalu kecil → penyebab utama) |
| Batch size            | 32                                     |
| Stage 1 val_acc akhir | 59.83%                                 |
| Stage 2 val_acc akhir | 73.14%                                 |
| Stage 3 val_acc akhir | 71.58%                                 |

### Genotype Akhir (COLLAPSED)

```
Normal Cell:
  node_2 = max_pool_3x3(node_0) + max_pool_3x3(node_1)
  node_3 = max_pool_3x3(node_2) + max_pool_3x3(node_0)
  node_4 = max_pool_3x3(node_3) + max_pool_3x3(node_2)
  node_5 = max_pool_3x3(node_3) + max_pool_3x3(node_2)

Reduce Cell:
  SEMUA = max_pool_3x3
```

**100% max_pool_3x3 di semua node, semua cell → ARCHITECTURE COLLAPSED**

---

## 2. Jejak Pruning Stage by Stage

### Stage 1 → Stage 2 (8 ops → 5 ops)

```
Input:   [none, skip_connect, sep_conv_3x3, sep_conv_5x5, dil_conv_3x3, dil_conv_5x5, avg_pool_3x3, max_pool_3x3]
Output:  [none, skip_connect, sep_conv_5x5, dil_conv_3x3, max_pool_3x3]
Dihapus: sep_conv_3x3, dil_conv_5x5, avg_pool_3x3
```

→ Masih ada 2 conv ops (sep_conv_5x5, dil_conv_3x3), masih aman.

### Stage 2 → Stage 3 (5 ops → 3 ops) ← KRITIS

```
Input:   [none, skip_connect, sep_conv_5x5, dil_conv_3x3, max_pool_3x3]
Output:  [none, skip_connect, max_pool_3x3]
Dihapus: sep_conv_5x5, dil_conv_3x3
```

→ **SEMUA conv ops dihapus!** Hanya sisa `none`, `skip_connect`, `max_pool_3x3`.

### Stage 3 (3 ops tersisa)

```
Primitives: [none, skip_connect, max_pool_3x3]

Alpha distribution (Node 2):
  edge(0→2): none=0.071  skip=0.061  max_pool=0.868  ← 87% max_pool
  edge(1→2): none=0.479  skip=0.061  max_pool=0.460
```

→ Tanpa pilihan conv, `max_pool_3x3` mendominasi secara otomatis.

---

## 3. Root Cause Analysis (Mengapa Terjadi)

### Penyebab Utama: C_search = 8 (Terlalu Kecil)

**Mekanisme collapse:**

```
C_search=8 → Supernet sangat sempit

Operasi conv (sep_conv, dil_conv):
  - Butuh representasi channel yang kaya untuk efektif
  - Dengan C=8, conv tidak mampu belajar fitur bermakna
  - Gradient terhadap alpha conv → kecil → alpha tidak naik

Operasi parameter-free (max_pool, none, skip_connect):
  - TIDAK membutuhkan channel width
  - Bekerja sama baik di C=8 maupun C=64
  - Alpha mereka relatif lebih tinggi
  - DARTS α optimizer memilih mereka secara konsisten
```

**Ilustrasi ketidakseimbangan alpha:**

| Operasi      | Tipe           | C_search=8                         | C_search=16                         |
| ------------ | -------------- | ---------------------------------- | ----------------------------------- |
| dil_conv_3x3 | Parameterized  | Underfed, alpha rendah             | Belajar baik, alpha kompetitif      |
| sep_conv_3x3 | Parameterized  | Underfed, alpha rendah             | Belajar baik, alpha kompetitif      |
| max_pool_3x3 | Parameter-free | Alpha tinggi (tidak terpengaruh C) | Alpha kompetitif, tidak dominan     |
| skip_connect | Parameter-free | Alpha tinggi                       | Alpha cukup, dikontrol skip_dropout |
| none         | Parameter-free | Alpha tinggi (zero = safe)         | Difilter oleh STRUCTURAL_OPS        |

### Penyebab Sekunder: STRUCTURAL_OPS Menyertakan 'none'

Pada saat Run #1, kode `prune_operations()` menggunakan:

```python
STRUCTURAL_OPS = {'none', 'skip_connect'}  # keduanya auto-kept
MIN_CONV = 1  # hanya 1 conv yang dijamin
```

Dengan 3 slots di Stage 3:

- Slot 1: `none` (auto-kept oleh STRUCTURAL_OPS)
- Slot 2: `skip_connect` (auto-kept oleh STRUCTURAL_OPS)
- Slot 3: 1 slot sisa → diberikan ke ops dengan alpha tertinggi = `max_pool_3x3`

→ Conv ops tidak mendapat slot sama sekali.

### Penyebab Tersier: Pruning Stage 2→3 Terlalu Agresif

Stage 2 masih punya: `[none, skip_connect, sep_conv_5x5, dil_conv_3x3, max_pool_3x3]`

Alpha Stage 2 Node 3, edge(0→3):

```
none=0.154  skip=0.085  sep_conv_5=0.041  dil_conv=0.044  max_pool=0.676
```

→ max_pool_3x3 alpha sudah sangat dominan (0.676) karena C_search=8.
→ Ketika dipotong jadi 3, conv menjadi korban pertama.

---

## 4. Timeline Collapse

```
Stage 1 (C_search=8, 5 cells, 50 epochs):
  Epoch 1:  val_acc=0.48%  (random, semua alpha uniform)
  Epoch 15: val_acc=5.16%  (α mulai diverge, warmup selesai)
  Epoch 20: val_acc=20.38% (α aktif diupdate, max_pool naik)
  Epoch 50: val_acc=59.83%
  → Stage 1 Genotype: Normal masih MIXED (max_pool + dil_conv + sep_conv)
                                                    ↑ masih ada conv!

Stage 1→2 Pruning:
  Dihapus: sep_conv_3x3, dil_conv_5x5, avg_pool_3x3
  Sisa: none, skip_connect, sep_conv_5x5, dil_conv_3x3, max_pool_3x3
  → Masih 2 conv ops, belum collapse

Stage 2 (C_search=8, 8 cells, 50 epochs):
  Alpha max_pool terus naik karena C_search masih 8
  Stage 2 Normal dominant: max_pool_3x3 (6 dari 8 operasi)
  Epoch 50 val_acc=73.14%

Stage 2→3 Pruning: ← TITIK COLLAPSE
  Dihapus: sep_conv_5x5, dil_conv_3x3 (alpha mereka kalah dari max_pool)
  Sisa: none, skip_connect, max_pool_3x3
  → TIDAK ADA conv ops tersisa!

Stage 3 (C_search=8, 11 cells, 50 epochs):
  Hanya: [none, skip_connect, max_pool_3x3]
  max_pool alpha avg = 0.87 → dominasi total
  Epoch 50 val_acc=71.58% (turun dari Stage 2 karena architecture memburuk)
```

---

## 5. Tanda-Tanda Peringatan yang Terlewat

| Indikator                    | Stage 1                    | Stage 2                   | Stage 3                 |
| ---------------------------- | -------------------------- | ------------------------- | ----------------------- |
| Dominasi max_pool alpha      | Mulai terlihat (0.27-0.68) | Sangat tinggi (0.68)      | 0.87 (fatal)            |
| Conv alpha Stage 2 edge(0→3) | —                          | sep_conv=0.041, dil=0.044 | ❌ Sudah terlalu rendah |
| Val_acc trend                | ↑ 59.83%                   | ↑ 73.14%                  | ↓ 71.58%                |
| Stage 3 ops diversity        | Banyak                     | Berkurang                 | **Nol conv**            |

---

## 6. Fix yang Diterapkan untuk Run #4

### Fix 1: C_search ditingkatkan (8 → 16)

```python
# nas_config.py
"C_search": 16  # dari 8
"batch_size": 16  # dari 32, menyesuaikan VRAM
```

### Fix 2: STRUCTURAL_OPS — hapus 'none'

```python
# search.py - prune_operations()
STRUCTURAL_OPS = {'skip_connect'}  # 'none' dihapus
# 'none' sekarang harus bersaing secara kompetitif
```

### Fix 3: MIN_CONV ditingkatkan (1 → 2)

```python
# search.py - prune_operations()
MIN_CONV = 2  # dari 1
# Memastikan minimal 2 conv ops berbeda survive ke Stage 3
```

**Hasil expected Run #4:**

```
Stage 3 ops: [skip_connect, dil_conv_3x3, sep_conv_3x3]  ← minimum
             atau variasi lain dengan 2 conv ops
BUKAN:       [none, skip_connect, max_pool_3x3]  ← Run #1 (bad)
```

---

## 7. Perbandingan Run #1 vs Run #3

| Aspek                | Run #1 (Collapsed)     | Run #3 (Low Diversity)               |
| -------------------- | ---------------------- | ------------------------------------ |
| C_search             | 8                      | 16                                   |
| Batch size           | 32                     | 16                                   |
| Stage 3 ops          | none, skip, max_pool   | skip, dil_conv×3                     |
| Normal Cell ops      | 100% max_pool          | 100% dil_conv_3x3                    |
| Search val_acc       | 71.58%                 | 89.5%                                |
| Architecture quality | **UNUSABLE**           | Usable (retrain 98.44%)              |
| Root cause           | C_search terlalu kecil | MIN_CONV=1, 'none' di STRUCTURAL_OPS |

---

## 8. Kesimpulan

Run #1 mengalami **architecture collapse** yang disebabkan oleh kombinasi:

1. **C_search=8** → conv ops tidak bisa belajar efektif → alpha conv lemah
2. **STRUCTURAL_OPS={'none','skip_connect'}** → membuang 2 dari 3 slot Stage 3 untuk ops non-conv
3. **MIN_CONV=1** → hanya 1 conv dijamin, tidak cukup diversitas

Semua 3 faktor berkonspirasi memaksa `max_pool_3x3` mengisi semua slot di Stage 3.  
Architecture yang dihasilkan adalah **non-functional** untuk knowledge distillation.

Genotype ini **tidak digunakan** untuk retrain dan **tidak digunakan** sebagai student model.

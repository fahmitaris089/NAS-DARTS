# Illumination Preprocessing Diagnostic

- Student weights: `knowledge_distilation/kd_results/kd_L0.05_C12_cells10_stem8_t20_a05_lr1e4_e160_nw0/best_model.pth`
- Test samples: `834`
- Best method: `none` = 99.76% (832/834)

## Results

| Method | Accuracy | Correct | Delta vs none |
|---|---:|---:|---:|
| none | 99.76% | 832/834 | +0.00 pp |
| autocontrast | 99.76% | 832/834 | +0.00 pp |
| equalize | 7.55% | 63/834 | -92.21 pp |
| gamma_0.8 | 96.88% | 808/834 | -2.88 pp |
| gamma_0.9 | 99.40% | 829/834 | -0.36 pp |
| gamma_1.1 | 99.52% | 830/834 | -0.24 pp |
| gamma_1.2 | 98.56% | 822/834 | -1.20 pp |
| clahe_1.5 | 6.24% | 52/834 | -93.53 pp |
| clahe_2.0 | 3.00% | 25/834 | -96.76 pp |
| clahe_3.0 | 0.72% | 6/834 | -99.04 pp |

## Focus Errors

### 277_6.bmp

- `none`: pred=42 correct=False true_rank=2 true_prob=0.060025
- `autocontrast`: pred=42 correct=False true_rank=2 true_prob=0.060025
- `equalize`: pred=239 correct=False true_rank=262 true_prob=0.000933
- `gamma_0.8`: pred=42 correct=False true_rank=2 true_prob=0.124379
- `gamma_0.9`: pred=42 correct=False true_rank=2 true_prob=0.086502
- `gamma_1.1`: pred=42 correct=False true_rank=4 true_prob=0.037484
- `gamma_1.2`: pred=251 correct=False true_rank=7 true_prob=0.026137
- `clahe_1.5`: pred=470 correct=False true_rank=279 true_prob=0.000845
- `clahe_2.0`: pred=498 correct=False true_rank=321 true_prob=0.000611
- `clahe_3.0`: pred=454 correct=False true_rank=262 true_prob=0.000648

### 504_4.bmp

- `none`: pred=482 correct=False true_rank=31 true_prob=0.005012
- `autocontrast`: pred=482 correct=False true_rank=29 true_prob=0.004988
- `equalize`: pred=444 correct=False true_rank=165 true_prob=0.000901
- `gamma_0.8`: pred=482 correct=False true_rank=48 true_prob=0.003062
- `gamma_0.9`: pred=482 correct=False true_rank=40 true_prob=0.003916
- `gamma_1.1`: pred=468 correct=False true_rank=27 true_prob=0.005814
- `gamma_1.2`: pred=468 correct=False true_rank=24 true_prob=0.006231
- `clahe_1.5`: pred=351 correct=False true_rank=44 true_prob=0.003928
- `clahe_2.0`: pred=444 correct=False true_rank=37 true_prob=0.004033
- `clahe_3.0`: pred=444 correct=False true_rank=107 true_prob=0.000770

Contact sheet: `focus_preprocess_contact_sheet.png`
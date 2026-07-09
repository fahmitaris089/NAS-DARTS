# Preprocessing / Illumination Audit

- Errors CSV: `analysis/prediction_overlap_L005_C12_cells10_HintonKD_9976/c12_errors.csv`
- Data dir: `preprocessed_results`

## 277_6.bmp

- True subject: `277`
- C12 top-5 labels: `42|276|329|251|432`
- Compared subjects: `277, 43, 330`
- Contact sheet: `contact_277_6_true_277_vs_43_330.png`
- Diagnostic tags: `illumination_high_z2.15, overexposure_high_z3.00, vein_edge_strength_low_z-2.11`

| Metric | Focus | True class mean | z-score |
|---|---:|---:|---:|
| foreground_mean | 190.4335 | 167.9478 | 2.15 |
| foreground_std | 30.3323 | 30.7158 | -0.70 |
| high_pixel_ratio | 0.0266 | 0.0031 | 3.00 |
| foreground_ratio | 0.7942 | 0.6585 | 2.34 |
| bbox_area | 0.9955 | 0.9756 | 0.98 |
| center_x | 0.4978 | 0.5040 | -0.69 |
| center_y | 0.4955 | 0.4917 | 0.89 |
| margin_top | 0.0000 | 0.0000 | 0.00 |
| gradient_mean | 0.0228 | 0.0263 | -2.11 |

## 504_4.bmp

- True subject: `504`
- C12 top-5 labels: `482|468|484|380|218`
- Compared subjects: `504, 483, 469, 485`
- Contact sheet: `contact_504_4_true_504_vs_483_469_485.png`
- Diagnostic tags: `illumination_high_z2.17, overexposure_high_z3.00, vein_edge_strength_low_z-1.73`

| Metric | Focus | True class mean | z-score |
|---|---:|---:|---:|
| foreground_mean | 198.7513 | 167.0316 | 2.17 |
| foreground_std | 24.5948 | 27.1019 | -0.88 |
| high_pixel_ratio | 0.0171 | 0.0019 | 3.00 |
| foreground_ratio | 0.8991 | 0.7543 | 2.29 |
| bbox_area | 1.0000 | 0.9781 | 0.95 |
| center_x | 0.4978 | 0.5087 | -0.95 |
| center_y | 0.4978 | 0.4978 | 0.00 |
| margin_top | 0.0000 | 0.0000 | 0.00 |
| gradient_mean | 0.0231 | 0.0255 | -1.73 |

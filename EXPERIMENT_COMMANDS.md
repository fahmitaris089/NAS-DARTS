# Experiment Commands - NAS, KD, Metric Fine-Tune

Dokumen ini merangkum command eksperimen yang sudah/perlu dicoba untuk membandingkan NAS baseline, retrain C12 stem8, beberapa varian KD, augmentasi, dan metric fine-tune. Semua command diasumsikan dijalankan dari root repo:

```powershell
cd C:\Users\Nanik Suciati\Downloads\NAS-DARTS
```

## 1. Baseline Path dan Kandidat Utama

Kandidat utama saat ini:

```text
nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls
```

Ringkasan hasil penting:

```text
C12 L0.05 stem8 FP32 test: 99.40% (829/834)
INT8 Pi: 99.28% (828/834)
MobileNetV3Small: 99.64% (831/834)
EfficientNetV2M teacher: 100.00%
ResNet50 teacher: 100.00%
```

Target eksperimen:

```text
Minimal: 99.52% (830/834)
Ideal  : 99.64% (831/834), setara/menang dari MobileNetV3Small
```

## 2. NAS Search Hardware-Aware

Search lama yang sudah ada:

```text
nas_results/search_hwint8_l0.05/genotype_final.json
nas_results/search_hwint8_l0.10/genotype_final.json
nas_results/search_hwint8_l0.20/genotype_final.json
```

Search ulang yang disarankan, tanpa mengubah search space:

```powershell
python search.py --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/search_hwint8_l0.03_seed42" --batch_size 16 --search_input_size 112 --oplat_lambda 0.03 --latency_lut "latency_lut_pi_int8_corrected.json" --seed 42
```

```powershell
python search.py --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/search_hwint8_l0.075_seed42" --batch_size 16 --search_input_size 112 --oplat_lambda 0.075 --latency_lut "latency_lut_pi_int8_corrected.json" --seed 42
```

Catatan:

- Tidak perlu rerun `L0.10` jika `nas_results/search_hwint8_l0.10/genotype_final.json` sudah valid.
- `palm_vein_dataset.py` update PK sampler tidak memengaruhi NAS search karena `search.py` memakai `create_search_dataloaders()`, bukan `create_retrain_dataloaders()`.
- Search tetap memakai random split search train/search val dari train split.

## 3. NAS Retrain Hasil Search

Default retrain yang paling apple-to-apple untuk kandidat baru:

```text
C_init=12
num_cells=8
stem_downsample=8
reduction_indices=2,5
epochs=300
batch_size=64
lr=0.001
weight_decay=0.05
drop_path_prob=0.2
cutout_length=16
augmentation_policy=v1_legacy
```

### 3.1 Retrain C12 Stem8 - Kandidat Utama

Retrain dari genotype L0.03 baru:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.03_seed42/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.03_C12_stemds8_834cls" --C_init 12 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

Retrain dari genotype L0.075 baru:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.075_seed42/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.075_C12_stemds8_834cls" --C_init 12 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

Retrain dari genotype L0.10 existing:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.10/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.10_C12_stemds8_834cls" --C_init 12 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

Retrain dari genotype L0.20 existing, jika ingin cek kandidat paling hardware-aware:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.20/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.20_C12_stemds8_834cls" --C_init 12 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

### 3.2 Retrain C10 Stem8 - Kapasitas Sedang

Gunakan C10 jika C12 latency terlalu tinggi, atau untuk membandingkan kapasitas sedang dengan C12.

L0.05 C10 stem8:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.05/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.05_C10_stemds8_834cls" --C_init 10 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

L0.10 C10 stem8:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.10/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.10_C10_stemds8_834cls" --C_init 10 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

L0.20 C10 stem8:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.20/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.20_C10_stemds8_834cls" --C_init 10 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

### 3.3 Retrain C8 Stem8 - Ringan dan Cepat

Gunakan C8 stem8 untuk kandidat yang harus sangat ringan. Jika akurasi < 99.0%, biasanya tidak perlu lanjut KD.

L0.05 C8 stem8:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.05/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.05_C8_stemds8_834cls" --C_init 8 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

L0.10 C8 stem8:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.10/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.10_C8_stemds8_834cls" --C_init 8 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

L0.20 C8 stem8:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.20/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.20_C8_stemds8_834cls" --C_init 8 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

### 3.4 Retrain Stem4 - Hanya Untuk Ablasi

Stem4 biasanya menaikkan FLOPs/latency di Raspberry Pi, jadi jangan jadi kandidat utama. Gunakan hanya untuk ablation study bahwa stem8 memang lebih efisien.

L0.05 C8 stem4:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.05/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.05_C8_stemds4_834cls" --C_init 8 --num_cells 8 --stem_downsample 4 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

L0.10 C8 stem4:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.10/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.10_C8_stemds4_834cls" --C_init 8 --num_cells 8 --stem_downsample 4 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

L0.20 C8 stem4:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.20/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.20_C8_stemds4_834cls" --C_init 8 --num_cells 8 --stem_downsample 4 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

### 3.5 Multi-Seed Retrain Untuk Kandidat Terbaik

Gunakan hanya setelah ada satu genotype yang kuat, misalnya L0.05 C12 stem8 atau L0.03 C12 stem8. Tujuannya mencari variasi training seed yang mungkin memperbaiki 1-2 sample residual.

Seed 7:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.05/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.05_C12_stemds8_seed7_834cls" --C_init 12 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy --seed 7
```

Seed 123:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.05/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.05_C12_stemds8_seed123_834cls" --C_init 12 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy --seed 123
```

Seed 2026:

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.05/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.05_C12_stemds8_seed2026_834cls" --C_init 12 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 300 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy --seed 2026
```

### 3.6 Quick Retrain 50 Epoch Untuk Screening

Jika GPU penuh dan hanya ingin cek apakah genotype berpotensi, gunakan 50 epoch dulu. Jangan jadikan hasil ini final tesis.

```powershell
python retrain.py --genotype "nas_results/search_hwint8_l0.03_seed42/genotype_final.json" --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --output_dir "nas_results/retrain_hwNAS_L0.03_C12_stemds8_screen50_834cls" --C_init 12 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 --epochs 50 --batch_size 64 --lr 0.001 --weight_decay 0.05 --drop_path_prob 0.2 --cutout_length 16 --augmentation_policy v1_legacy
```

Decision rule retrain:

```text
< 99.40%  : jangan lanjut KD/metric
= 99.40%  : cek overlap error; lanjut jika error set berbeda/lebih promising
>= 99.52% : export ONNX/INT8 dan benchmark Pi
```

Prioritas retrain yang disarankan:

```text
1. L0.03 C12 stem8
2. L0.075 C12 stem8
3. L0.10 existing C12 stem8
4. Multi-seed hanya untuk genotype yang sudah >= 99.40%
5. Stem4 hanya untuk ablation, bukan kandidat utama deployment
```

## 4. Export ONNX dan INT8

Contoh export kandidat C12:

```powershell
python export_kd_onnx_int8.py --model_dir "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls"
```

Benchmark Pi FP32 vs INT8:

```powershell
python benchmark_fp32_vs_int8_pi.py --model_dir "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls" --data_dir ".\preprocessed_results" --split_path ".\split_info.json"
```

Catatan:

- PTQ/INT8 tidak boleh dijadikan solusi utama menaikkan akurasi.
- FP32/PyTorch checkpoint harus sudah kuat lebih dulu.
- INT8 dipakai sebagai optimasi size/latency.

## 5. KD Hinton Dasar

KD C10 L0.05 yang pernah dites:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C10_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C10_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --temperature 3 --alpha 0.85 --label_smoothing 0.0 --no_mix --epochs 150 --batch_size 64 --lr 1e-4 --output_dir "knowledge_distilation/kd_results/kd_hwNAS_L0.05_C10_t3_a0.85_ls0_nomix_lr1e4"
```

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C10_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C10_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --temperature 3 --alpha 0.8 --label_smoothing 0.0 --no_mix --epochs 150 --batch_size 64 --lr 1e-4 --output_dir "knowledge_distilation/kd_results/kd_hwNAS_L0.05_C10_t3_a0.8_ls0_nomix_lr1e4"
```

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C10_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C10_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --temperature 4 --alpha 0.85 --label_smoothing 0.0 --no_mix --epochs 150 --batch_size 64 --lr 1e-4 --output_dir "knowledge_distilation/kd_results/kd_hwNAS_L0.05_C10_t4_a0.85_ls0_nomix_lr1e4"
```

KD C12 konservatif dengan EfficientNet:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --teacher_weights "Teacher/training_results/EfficientNetV2M/best_model.pth" --temperature 1 --alpha 0.995 --label_smoothing 0.0 --no_mix --epochs 80 --batch_size 64 --lr 1e-6 --weight_decay 1e-4 --drop_path 0.0 --freeze_bn --output_dir "knowledge_distilation/kd_results/kd_hwNAS_L0.05_C12_effv2m_t1_a0.995_lr1e6_wd1e4_dp0_freezebn"
```

KD C12 dengan ResNet50 teacher:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --teacher_arch resnet50 --teacher_weights "Teacher/training_results/ResNet50/best_model.pth" --temperature 1 --alpha 0.995 --label_smoothing 0.0 --no_mix --epochs 80 --batch_size 64 --lr 1e-6 --weight_decay 1e-4 --drop_path 0.0 --freeze_bn --output_dir "knowledge_distilation/kd_results/kd_hwNAS_L0.05_C12_resnet50_t1_a0.995_lr1e6_wd1e4_dp0_freezebn"
```

## 6. KD Temperature Tinggi dan No CutOut

Validasi apakah KD gagal karena temperature terlalu rendah atau augmentasi terlalu agresif.

T10 alpha 0.90, no CutOut, augmentasi ringan:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --teacher_weights "Teacher/training_results/EfficientNetV2M/best_model.pth" --temperature 10 --alpha 0.90 --label_smoothing 0.0 --no_mix --epochs 60 --batch_size 64 --lr 5e-7 --lr_min 5e-8 --weight_decay 1e-4 --drop_path 0.0 --freeze_bn --cutout_length 0 --augmentation_policy v3_no_flip_light --output_dir "knowledge_distilation/kd_results/kd_C12_effv2m_t10_a090_lr5e7_wd1e4_freezebn_v3light_nocutout"
```

T10 alpha 0.95, no CutOut:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --teacher_weights "Teacher/training_results/EfficientNetV2M/best_model.pth" --temperature 10 --alpha 0.95 --label_smoothing 0.0 --no_mix --epochs 60 --batch_size 64 --lr 5e-7 --lr_min 5e-8 --weight_decay 1e-4 --drop_path 0.0 --freeze_bn --cutout_length 0 --output_dir "knowledge_distilation/kd_results/kd_C12_effv2m_t10_a095_lr5e7_wd1e4_freezebn_nocutout"
```

T20 alpha 0.97, no CutOut:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --teacher_weights "Teacher/training_results/EfficientNetV2M/best_model.pth" --temperature 20 --alpha 0.97 --label_smoothing 0.0 --no_mix --epochs 60 --batch_size 64 --lr 5e-7 --lr_min 5e-8 --weight_decay 1e-4 --drop_path 0.0 --freeze_bn --cutout_length 0 --output_dir "knowledge_distilation/kd_results/kd_C12_effv2m_t20_a097_lr5e7_wd1e4_freezebn_nocutout"
```

Decision rule:

```text
Jika epoch 15-20 val_acc < 98.32%, stop.
Jika final tetap 99.40%, jalankan overlap analysis.
Jika 5 error tetap sama, tutup hipotesis temperature/augmentasi.
```

## 7. Pairwise / Embedding KD

Pairwise KD random sampler:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --teacher_weights "Teacher/training_results/EfficientNetV2M/best_model.pth" --kd_method pairwise --ce_weight 0.2 --relation_weight 2.0 --embedding_weight 0.0 --logit_kd_weight 0.0 --temperature 1 --label_smoothing 0.0 --no_mix --epochs 100 --batch_size 64 --train_sampler random --lr 1e-7 --lr_min 1e-8 --weight_decay 0.0 --drop_path 0.0 --freeze_bn --output_dir "knowledge_distilation/kd_results/kd_pairwise_C12_effv2m_ce02_rel2_lr1e7_min1e8_wd0_freezebn"
```

Pairwise KD dengan PK sampler:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --teacher_weights "Teacher/training_results/EfficientNetV2M/best_model.pth" --kd_method pairwise --ce_weight 0.2 --relation_weight 5.0 --embedding_weight 0.0 --logit_kd_weight 0.0 --temperature 1 --label_smoothing 0.0 --no_mix --epochs 80 --batch_size 64 --train_sampler pk --pk_p 16 --pk_k 4 --lr 2e-7 --lr_min 2e-8 --weight_decay 0.0 --drop_path 0.0 --freeze_bn --output_dir "knowledge_distilation/kd_results/kd_pairwise_PK16x4_C12_effv2m_ce02_rel5_lr2e7_min2e8_wd0_freezebn"
```

Catatan:

- Pairwise KD butuh batch dengan pasangan positif/negatif bermakna.
- `--train_sampler pk --pk_p 16 --pk_k 4` membuat batch 64 berisi 16 identitas x 4 sample.
- MixUp/CutMix harus off untuk relation/embedding KD.

## 8. Hard Top-K / Margin KD

Hard top-k konservatif:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --teacher_weights "Teacher/training_results/EfficientNetV2M/best_model.pth" --kd_method hard_topk --ce_weight 1.0 --topk_k 5 --topk_weight 0.05 --margin_weight 0.10 --margin_m 0.10 --hard_weight 2.0 --hard_margin_threshold 0.20 --teacher_conf_threshold 0.50 --temperature 2 --label_smoothing 0.0 --no_mix --epochs 80 --batch_size 64 --lr 5e-7 --lr_min 5e-8 --weight_decay 0.0 --drop_path 0.0 --freeze_bn --output_dir "knowledge_distilation/kd_results/kd_hardtopk_C12_effv2m_k5_tw005_mw010_m010_lr5e7_freezebn"
```

Hard top-k lebih kuat:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --teacher_weights "Teacher/training_results/EfficientNetV2M/best_model.pth" --kd_method hard_topk --ce_weight 1.0 --topk_k 5 --topk_weight 0.02 --margin_weight 0.20 --margin_m 3.0 --hard_weight 2.0 --hard_margin_threshold 5.0 --teacher_conf_threshold 0.50 --temperature 2 --label_smoothing 0.0 --no_mix --epochs 60 --batch_size 64 --lr 2e-7 --lr_min 2e-8 --weight_decay 0.0 --drop_path 0.0 --freeze_bn --output_dir "knowledge_distilation/kd_results/kd_hardtopk_C12_effv2m_k5_tw002_mw020_m3_hm5_lr2e7_freezebn"
```

Catatan hasil:

- Hard top-k bisa memperbaiki margin, tetapi pada eksperimen sebelumnya cenderung menambah error baru atau turun ke `99.28%`.
- Jika dipakai lagi, wajib cek overlap, bukan hanya akurasi.

## 9. Conservative KD dan Multi-Teacher KD

Conservative KD anchor:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --anchor_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --teacher_weights "Teacher/training_results/EfficientNetV2M/best_model.pth" --kd_method conservative --ce_weight 1.0 --topk_k 5 --topk_weight 0.03 --margin_weight 0.08 --margin_m 2.0 --anchor_weight 0.5 --anchor_temperature 2 --temperature 2 --label_smoothing 0.0 --no_mix --epochs 60 --batch_size 64 --lr 1e-7 --lr_min 1e-8 --weight_decay 0.0 --drop_path 0.0 --freeze_bn --output_dir "knowledge_distilation/kd_results/kd_conservative_C12_effv2m_anchor05_tw003_mw008_m2_lr1e7_freezebn"
```

Conservative multi-teacher EfficientNet + MobileNet:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --anchor_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --teacher_weights "Teacher/training_results/EfficientNetV2M/best_model.pth" --teacher2_arch mobilenet_v3_small --teacher2_weights "Teacher/training_results/MobileNetV3Small/best_model.pth" --kd_method conservative_multiteacher --ce_weight 1.0 --topk_k 5 --teacher1_weight 0.01 --teacher2_weight 0.05 --teacher2_conf_threshold 0.05 --teacher_agree_bonus 1.5 --teacher_disagree_policy teacher2_only --anchor_weight 0.5 --anchor_temperature 2 --temperature 2 --label_smoothing 0.0 --no_mix --epochs 60 --batch_size 64 --lr 1e-7 --lr_min 1e-8 --weight_decay 0.0 --drop_path 0.0 --freeze_bn --output_dir "knowledge_distilation/kd_results/kd_consMT_C12_effv2m_mbv3_anchor05_t1w001_t2w005_lr1e7_freezebn"
```

Continue conservative multi-teacher:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "knowledge_distilation/kd_results/kd_consMT_C12_effv2m_mbv3_anchor05_t1w001_t2w005_lr1e7_freezebn/best_model.pth" --anchor_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --teacher_weights "Teacher/training_results/EfficientNetV2M/best_model.pth" --teacher2_arch mobilenet_v3_small --teacher2_weights "Teacher/training_results/MobileNetV3Small/best_model.pth" --kd_method conservative_multiteacher --ce_weight 1.0 --topk_k 5 --teacher1_weight 0.01 --teacher2_weight 0.05 --teacher2_conf_threshold 0.05 --teacher_agree_bonus 1.5 --teacher_disagree_policy teacher2_only --anchor_weight 0.5 --anchor_temperature 2 --temperature 2 --label_smoothing 0.0 --no_mix --epochs 60 --batch_size 64 --lr 5e-8 --lr_min 1e-8 --weight_decay 0.0 --drop_path 0.0 --freeze_bn --output_dir "knowledge_distilation/kd_results/kd_consMT_C12_effv2m_mbv3_anchor05_continue_lr5e8_freezebn"
```

Catatan hasil:

- Conservative KD menjaga agar model tidak collapse.
- Multi-teacher tetap mentok di `99.40%` pada hasil sebelumnya.
- Jika error overlap tetap sama, jangan tambah epoch sampai 300; ubah metode atau cari genotype baru.

## 10. Metric Fine-Tune ArcFace

ArcFace v1, agresif, pernah turun:

```powershell
python metric_finetune.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --method arcface --ce_weight 0.5 --metric_weight 0.5 --arc_s 16 --arc_m 0.2 --epochs 80 --batch_size 64 --lr 5e-6 --lr_min 5e-7 --weight_decay 1e-4 --drop_path 0.0 --freeze_bn --cutout_length 16 --augmentation_policy v1_legacy --output_dir "metric_results/arcface_C12_L005_s16_m02_ce05_w05_lr5e6_freezebn"
```

ArcFace konservatif, no CutOut, juga pernah turun:

```powershell
python metric_finetune.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --method arcface --ce_weight 0.9 --metric_weight 0.1 --arc_s 8 --arc_m 0.05 --epochs 60 --batch_size 64 --lr 5e-7 --lr_min 5e-8 --weight_decay 0.0 --drop_path 0.0 --freeze_bn --cutout_length 0 --augmentation_policy v1_legacy --output_dir "metric_results/arcface_C12_L005_s8_m005_ce09_w01_lr5e7_nocutout_freezebn"
```

Catatan hasil:

- ArcFace membuat CE/train makin bagus, tetapi val/test turun.
- Untuk sementara jangan lanjut ArcFace.
- Jika metric-learning dilanjutkan, arah berikutnya lebih masuk akal: SupCon + CE dengan PK sampler.

## 11. Augmentation Policy

Policy tersedia di `palm_vein_dataset.py`:

```text
v1_legacy
- Resize
- RandomHorizontalFlip(p=0.5)
- RandomRotation(10)
- RandomAffine translate 0.05, scale 0.95-1.05
- ColorJitter brightness 0.15, contrast 0.10
- optional CutOut
```

```text
v2_multi_distance
- Resize
- No horizontal flip
- RandomRotation(15)
- RandomAffine translate 0.08, scale 0.78-1.28
- ColorJitter brightness 0.20, contrast 0.15
- optional CutOut
```

```text
v3_no_flip_light
- Resize
- No horizontal flip
- RandomRotation(5)
- RandomAffine translate 0.03, scale 0.97-1.08
- ColorJitter brightness 0.08, contrast 0.05
- optional CutOut
```

Rekomendasi penggunaan:

```text
NAS search/retrain standar : v1_legacy, CutOut 16
KD konservatif/fine-tune   : v3_no_flip_light, CutOut 0
Pairwise/SupCon/metric     : v3_no_flip_light, CutOut 0, PK sampler
Multi-distance robustness  : v2_multi_distance, hati-hati karena lebih agresif
```

Contoh KD dengan augmentasi ringan:

```powershell
python knowledge_distilation/kd_train.py --student_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --student_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --teacher_arch efficientnet_v2_m --teacher_weights "Teacher/training_results/EfficientNetV2M/best_model.pth" --temperature 10 --alpha 0.90 --label_smoothing 0.0 --no_mix --epochs 60 --batch_size 64 --lr 5e-7 --lr_min 5e-8 --weight_decay 1e-4 --drop_path 0.0 --freeze_bn --cutout_length 0 --augmentation_policy v3_no_flip_light --output_dir "knowledge_distilation/kd_results/kd_C12_effv2m_t10_a090_lr5e7_wd1e4_freezebn_v3light_nocutout"
```

## 12. Prediction Overlap Analysis

Analisis original C12:

```powershell
python analyze_prediction_overlap.py --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --nas_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --nas_weights "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/best_model.pth" --output_dir "analysis/prediction_overlap_C12"
```

Analisis hasil KD/metric:

```powershell
python analyze_prediction_overlap.py --data_dir ".\preprocessed_results" --split_path ".\split_info.json" --nas_config "nas_results/retrain_hwNAS_L0.05_C12_stemds8_834cls/config.json" --nas_weights "PATH\TO\RUN\best_model.pth" --output_dir "analysis/prediction_overlap_NAMA_RUN"
```

Interpretasi utama:

```text
Jika akurasi tetap 99.40% tapi error set berubah, run masih informatif.
Jika error tetap sama persis, metode itu mentok.
Jika error lama berkurang tanpa error baru, run promising meskipun naik hanya 1 sample.
```

## 13. Catatan Decision Akhir

Prioritas eksperimen saat ini:

```text
1. KD T10 alpha 0.90 + v3_no_flip_light + no CutOut
2. NAS search ulang lambda 0.03 dan 0.075
3. Retrain genotype baru dengan C12 stem8
4. Jika KD/ArcFace tetap mentok, coba SupCon + CE dengan PK sampler
```

Jangan lanjutkan terlalu lama jika:

```text
val_acc turun konsisten di bawah 98.20%
test turun di bawah 99.40%
overlap error tetap sama setelah beberapa variasi KD
```

Layak dilanjutkan jika:

```text
test >= 99.52%
atau error overlap menunjukkan minimal 1 error lama benar tanpa error baru
atau genotype baru punya latency Pi tetap < MobileNetV3Small dengan akurasi >= 99.40%
```

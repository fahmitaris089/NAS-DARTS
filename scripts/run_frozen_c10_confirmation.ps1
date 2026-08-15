param(
  [Parameter(Mandatory=$true)]
  [ValidateSet("pk_ce","progressive_center","progressive_hybrid","ce_ls0","adaface","arcface","subcenter")]
  [string]$Method,
  [ValidateSet(10,12)]
  [int]$CInit = 10,
  [int[]]$Seeds = @(123,2026),
  [Parameter(Mandatory=$true)]
  [switch]$ConfirmMethodFrozen
)

$ErrorActionPreference = "Stop"
if ($Seeds -contains 42) { throw "Seed 42 is screening-only here; confirmation seeds are 123 and 2026" }
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$BaseConfig = if ($CInit -eq 12) { "nas_results/retrain_l020_c12_pk_ce_300e/seed_42/config.json" } else { "nas_results/retrain_hwint8_l020_c10_stem8_robust_300e/seed_42/config.json" }
if (-not (Test-Path $BaseConfig)) { throw "Frozen architecture config missing: $BaseConfig" }
$Genotype = "nas_results/search_hwint8_l0.20/genotype_final.json"
$Teacher = "Eksperimen_Hardware_Aware_PDARTS/checkpoints/teacher/EfficientNetV2M_best_model.pth"
$Data = "preprocessed_results"
$Split = "PalmVein_Lightweight_Benchmark/dataset/splits/split_info.json"

foreach ($Seed in $Seeds) {
  $Initial = "nas_results/controlled_initial_states/l020_c${CInit}_stem8_cells8_seed${Seed}.pth"
  & py -3.14 scripts/create_c10_initial_state.py --config $BaseConfig --output $Initial --seed $Seed --C-init $CInit
  if ($LASTEXITCODE -ne 0) { throw "Initial-state creation failed for seed $Seed" }
  if ($Method -in @("progressive_center","progressive_hybrid")) {
    $Relation = if ($Method -eq "progressive_hybrid") { "0.05" } else { "0" }
    $Output = "knowledge_distilation/kd_results/frozen_${Method}_l020_c${CInit}_seed${Seed}"
    & py -3.14 knowledge_distilation/kd_train.py --teacher_arch efficientnet_v2_m --teacher_weights $Teacher --student_config $BaseConfig --student_weights $Initial --initial_student_weights $Initial --no_pretrained_student --data_dir $Data --split_path $Split --kd_method adaptive_center_relation --progressive_staging --progressive_center_start 101 --progressive_relation_start 201 --progressive_calibration_batches 10 --progressive_center_grad_ratio 0.10 --progressive_feature_grad_ratio 0.05 --progressive_relation_grad_ratio 0.05 --epochs 300 --batch_size 64 --lr 0.001 --lr_min 0.000001 --weight_decay 0.05 --warmup_epochs 10 --augmentation_policy v4_robust_light --train_sampler pk --pk_p 16 --pk_k 4 --center_weight 0.5 --feature_weight 0.1 --relation_weight $Relation --adaptive_warmup_epochs 20 --label_smoothing 0.2 --drop_path 0 --cutout_length 0 --no_mix --seed $Seed --num_workers 0 --output_dir $Output --skip-test-evaluation
  } else {
    $LossMode = switch ($Method) { "pk_ce" { "ce" }; "ce_ls0" { "ce" }; "adaface" { "adaface" }; "arcface" { "arcface" }; "subcenter" { "subcenter_arcface" } }
    $LabelSmoothing = if ($Method -eq "pk_ce") { "0.2" } else { "0" }
    $Output = "nas_results/frozen_${Method}_l020_c${CInit}_300e/seed_${Seed}"
    & py -3.14 Eksperimen_Hardware_Aware_PDARTS/src/nas/retrain.py --genotype $Genotype --data_dir $Data --split_path $Split --output_dir $Output --C_init $CInit --num_cells 8 --stem_downsample 8 --reduction_indices "2,5" --epochs 300 --batch_size 64 --lr 0.001 --lr_min 0.000001 --weight_decay 0.05 --warmup_epochs 10 --drop_path_prob 0 --cutout_length 0 --augmentation_policy v4_robust_light --train_sampler pk --pk_p 16 --pk_k 4 --initial_weights $Initial --checkpoint_epochs 100 --no_auxiliary --loss-mode $LossMode --label-smoothing $LabelSmoothing --arcface-margin 0.5 --arcface-scale 64 --seed $Seed --num_workers 0 --skip-test-evaluation
  }
  if ($LASTEXITCODE -ne 0) { throw "Frozen confirmation failed: method=$Method seed=$Seed" }
}

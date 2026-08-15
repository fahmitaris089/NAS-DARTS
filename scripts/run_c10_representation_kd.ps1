param(
  [Parameter(Mandatory=$true)]
  [ValidateSet("audit","pk_ce","center","hybrid_scratch","hybrid_early")]
  [string]$Mode,
  [switch]$Smoke,
  [switch]$AllowLegacy
)

$ErrorActionPreference = "Stop"
if (-not $AllowLegacy) {
  throw "This is the legacy unbounded E1-E4 runner. Use scripts/run_c10_targeted_screening.ps1. Pass -AllowLegacy only to reproduce an already registered historical run."
}
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Config = "nas_results/retrain_hwint8_l020_c10_stem8_robust_300e/seed_42/config.json"
$Genotype = "nas_results/search_hwint8_l0.20/genotype_final.json"
$Initial = "nas_results/controlled_initial_states/l020_c10_stem8_cells8_seed42.pth"
$Teacher = "Eksperimen_Hardware_Aware_PDARTS/checkpoints/teacher/EfficientNetV2M_best_model.pth"
$Data = "preprocessed_results"
$Split = "PalmVein_Lightweight_Benchmark/dataset/splits/split_info.json"
$ScratchEpochs = if ($Smoke) { 1 } else { 300 }
$ContinuationEpochs = if ($Smoke) { 1 } else { 200 }
$Suffix = if ($Smoke) { "_smoke" } else { "" }

if ($Mode -eq "audit") {
  & py -3.14 knowledge_distilation/audit_validation_errors.py --config configs/c10_error_audit.json
  exit $LASTEXITCODE
}

& py -3.14 scripts/create_c10_initial_state.py --config $Config --output $Initial --seed 42
if ($LASTEXITCODE -ne 0) { throw "Failed to create/reuse controlled initial state" }

if ($Mode -eq "pk_ce") {
  $E1Output = "nas_results/retrain_l020_c10_pk_ce_300e$Suffix/seed_42"
  $CheckpointEpoch = if ($Smoke) { 1 } else { 100 }
  & py -3.14 Eksperimen_Hardware_Aware_PDARTS/src/nas/retrain.py --genotype $Genotype --data_dir $Data --split_path $Split --output_dir $E1Output --C_init 10 --num_cells 8 --stem_downsample 8 --reduction_indices "2,5" --epochs $ScratchEpochs --batch_size 64 --lr 0.001 --lr_min 0.000001 --weight_decay 0.05 --warmup_epochs 10 --drop_path_prob 0 --cutout_length 0 --augmentation_policy v4_robust_light --train_sampler pk --pk_p 16 --pk_k 4 --initial_weights $Initial --checkpoint_epochs $CheckpointEpoch --no_auxiliary --seed 42 --num_workers 0 --skip-test-evaluation
} elseif ($Mode -eq "center" -or $Mode -eq "hybrid_scratch") {
  $Relation = if ($Mode -eq "center") { "0" } else { "0.05" }
  $Output = if ($Mode -eq "center") { "knowledge_distilation/kd_results/center_l020_c10_pk_scratch_seed42$Suffix" } else { "knowledge_distilation/kd_results/hybrid_l020_c10_pk_scratch_seed42$Suffix" }
  & py -3.14 knowledge_distilation/kd_train.py --teacher_arch efficientnet_v2_m --teacher_weights $Teacher --student_config $Config --student_weights $Initial --initial_student_weights $Initial --no_pretrained_student --data_dir $Data --split_path $Split --kd_method adaptive_center_relation --epochs $ScratchEpochs --batch_size 64 --lr 0.001 --lr_min 0.000001 --weight_decay 0.05 --warmup_epochs 10 --augmentation_policy v4_robust_light --train_sampler pk --pk_p 16 --pk_k 4 --center_weight 0.5 --feature_weight 0.1 --relation_weight $Relation --center_scale 64 --center_margin 0.35 --relation_topk 8 --relation_difference_threshold 0.02 --adaptive_warmup_epochs 20 --label_smoothing 0.2 --drop_path 0 --cutout_length 0 --no_mix --seed 42 --num_workers 0 --output_dir $Output --skip-test-evaluation
} else {
  $Early = "nas_results/retrain_l020_c10_pk_ce_300e/seed_42/checkpoints/epoch_100.pth"
  if (-not (Test-Path $Early)) { throw "Run -Mode pk_ce first; missing $Early" }
  & py -3.14 knowledge_distilation/kd_train.py --teacher_arch efficientnet_v2_m --teacher_weights $Teacher --student_config $Config --student_weights $Early --data_dir $Data --split_path $Split --kd_method adaptive_center_relation --epochs $ContinuationEpochs --batch_size 64 --lr 0.0003 --lr_min 0.000001 --weight_decay 0.02 --warmup_epochs 5 --augmentation_policy v4_robust_light --train_sampler pk --pk_p 16 --pk_k 4 --center_weight 0.5 --feature_weight 0.1 --relation_weight 0.05 --center_scale 64 --center_margin 0.35 --relation_topk 8 --relation_difference_threshold 0.02 --adaptive_warmup_epochs 20 --label_smoothing 0.2 --drop_path 0 --cutout_length 0 --no_mix --continuation_type weights_only --continuation_source_epoch 100 --seed 42 --num_workers 0 --output_dir "knowledge_distilation/kd_results/hybrid_l020_c10_pk_early100_seed42$Suffix" --skip-test-evaluation
}
if ($LASTEXITCODE -ne 0) { throw "Experiment '$Mode' failed with exit code $LASTEXITCODE" }

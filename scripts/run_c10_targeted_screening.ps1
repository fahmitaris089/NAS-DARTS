param(
  [Parameter(Mandatory=$true)]
  [ValidateSet("forensics","progressive_center","progressive_hybrid","ce_ls0","adaface","arcface","subcenter","c12_pk_ce","c12_branch")]
  [string]$Mode,
  [switch]$Smoke,
  [switch]$ForceBranch,
  [ValidateSet("progressive_center","progressive_hybrid","adaface","arcface","subcenter")]
  [string]$C12Method = "progressive_hybrid"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Config = "nas_results/retrain_hwint8_l020_c10_stem8_robust_300e/seed_42/config.json"
$Genotype = "nas_results/search_hwint8_l0.20/genotype_final.json"
$Initial = "nas_results/controlled_initial_states/l020_c10_stem8_cells8_seed42.pth"
$Teacher = "Eksperimen_Hardware_Aware_PDARTS/checkpoints/teacher/EfficientNetV2M_best_model.pth"
$Data = "preprocessed_results"
$Split = "PalmVein_Lightweight_Benchmark/dataset/splits/split_info.json"
$Decision = "results/diagnostics/c10_error_forensics_seed42/decision_gate.json"
$Epochs = if ($Smoke) { 3 } else { 300 }
$CenterStart = if ($Smoke) { 2 } else { 101 }
$RelationStart = if ($Smoke) { 3 } else { 201 }
$CalibrationBatches = if ($Smoke) { 1 } else { 10 }
$AdaptiveRamp = if ($Smoke) { 1 } else { 20 }
$Warmup = if ($Smoke) { 1 } else { 10 }
$CheckpointEpoch = if ($Smoke) { 1 } else { 100 }
$Suffix = if ($Smoke) { "_smoke" } else { "" }

if ($Mode -eq "forensics") {
  & py -3.14 knowledge_distilation/audit_validation_errors.py --config configs/c10_error_audit.json
  exit $LASTEXITCODE
}
if (-not (Test-Path $Decision)) { throw "Run -Mode forensics first; missing $Decision" }
$Gate = (Get-Content $Decision -Raw | ConvertFrom-Json).recommended_branch
$Allowed = @{
  "progressive_center_relation" = @("progressive_center","progressive_hybrid")
  "adaface_matched_control" = @("ce_ls0","adaface")
  "arcface_vs_subcenter" = @("arcface","subcenter")
  "ensemble_and_weight_soup_diagnostic" = @()
  "data_roi_label_audit" = @()
  "stop_no_supported_branch" = @()
}
if (-not $ForceBranch -and $Mode -notin @("c12_pk_ce","c12_branch") -and $Allowed[$Gate] -notcontains $Mode) {
  throw "Forensics selected '$Gate'; mode '$Mode' is not permitted. Use -ForceBranch only with a documented scientific reason."
}

& py -3.14 scripts/create_c10_initial_state.py --config $Config --output $Initial --seed 42
if ($LASTEXITCODE -ne 0) { throw "Failed to create/reuse controlled C10 initial state" }
$ProtocolTag = if ($Mode -in @("arcface","subcenter")) { "-stabilized" } else { "" }
$RunId = "$Mode$ProtocolTag-seed42$Suffix"
if (-not $Smoke) {
  & py -3.14 scripts/c10_screening_ledger.py --run-id $RunId --branch $Gate --seed 42 --status reserve
  if ($LASTEXITCODE -ne 0) { throw "Bounded-run ledger rejected $RunId" }
}

try {
  if ($Mode -in @("progressive_center","progressive_hybrid")) {
    $Relation = if ($Mode -eq "progressive_hybrid") { "0.05" } else { "0" }
    $Output = "knowledge_distilation/kd_results/${Mode}_l020_c10_seed42$Suffix"
    & py -3.14 knowledge_distilation/kd_train.py --teacher_arch efficientnet_v2_m --teacher_weights $Teacher --student_config $Config --student_weights $Initial --initial_student_weights $Initial --no_pretrained_student --data_dir $Data --split_path $Split --kd_method adaptive_center_relation --progressive_staging --progressive_center_start $CenterStart --progressive_relation_start $RelationStart --progressive_calibration_batches $CalibrationBatches --progressive_center_grad_ratio 0.10 --progressive_feature_grad_ratio 0.05 --progressive_relation_grad_ratio 0.05 --epochs $Epochs --batch_size 64 --lr 0.001 --lr_min 0.000001 --weight_decay 0.05 --warmup_epochs $Warmup --augmentation_policy v4_robust_light --train_sampler pk --pk_p 16 --pk_k 4 --center_weight 0.5 --feature_weight 0.1 --relation_weight $Relation --adaptive_warmup_epochs $AdaptiveRamp --label_smoothing 0.2 --drop_path 0 --cutout_length 0 --no_mix --seed 42 --num_workers 0 --output_dir $Output --skip-test-evaluation
  } elseif ($Mode -eq "c12_pk_ce") {
    $Initial12 = "nas_results/controlled_initial_states/l020_c12_stem8_cells8_seed42.pth"
    & py -3.14 scripts/create_c10_initial_state.py --config $Config --output $Initial12 --seed 42 --C-init 12
    if ($LASTEXITCODE -ne 0) { throw "Failed to create/reuse C12 initial state" }
    & py -3.14 Eksperimen_Hardware_Aware_PDARTS/src/nas/retrain.py --genotype $Genotype --data_dir $Data --split_path $Split --output_dir "nas_results/retrain_l020_c12_pk_ce_300e$Suffix/seed_42" --C_init 12 --num_cells 8 --stem_downsample 8 --reduction_indices "2,5" --epochs $Epochs --batch_size 64 --lr 0.001 --lr_min 0.000001 --weight_decay 0.05 --warmup_epochs $Warmup --drop_path_prob 0 --cutout_length 0 --augmentation_policy v4_robust_light --train_sampler pk --pk_p 16 --pk_k 4 --initial_weights $Initial12 --checkpoint_epochs $CheckpointEpoch --no_auxiliary --label-smoothing 0.2 --seed 42 --num_workers 0 --skip-test-evaluation
  } elseif ($Mode -eq "c12_branch") {
    $C12Gate = "results/diagnostics/c12_capacity_gate.json"
    if (-not (Test-Path $C12Gate)) { throw "Run scripts/compare_c12_gate.py after c12_pk_ce; missing $C12Gate" }
    if (-not (Get-Content $C12Gate -Raw | ConvertFrom-Json).allow_method_run) {
      throw "C12 PK-CE did not pass the predeclared capacity gate; run 4 is prohibited"
    }
    $Initial12 = "nas_results/controlled_initial_states/l020_c12_stem8_cells8_seed42.pth"
    $C12Config = "nas_results/retrain_l020_c12_pk_ce_300e/seed_42/config.json"
    if ($C12Method -in @("progressive_center","progressive_hybrid")) {
      $Relation = if ($C12Method -eq "progressive_hybrid") { "0.05" } else { "0" }
      & py -3.14 knowledge_distilation/kd_train.py --teacher_arch efficientnet_v2_m --teacher_weights $Teacher --student_config $C12Config --student_weights $Initial12 --initial_student_weights $Initial12 --no_pretrained_student --data_dir $Data --split_path $Split --kd_method adaptive_center_relation --progressive_staging --progressive_center_start $CenterStart --progressive_relation_start $RelationStart --progressive_calibration_batches $CalibrationBatches --progressive_center_grad_ratio 0.10 --progressive_feature_grad_ratio 0.05 --progressive_relation_grad_ratio 0.05 --epochs $Epochs --batch_size 64 --lr 0.001 --lr_min 0.000001 --weight_decay 0.05 --warmup_epochs $Warmup --augmentation_policy v4_robust_light --train_sampler pk --pk_p 16 --pk_k 4 --center_weight 0.5 --feature_weight 0.1 --relation_weight $Relation --adaptive_warmup_epochs $AdaptiveRamp --label_smoothing 0.2 --drop_path 0 --cutout_length 0 --no_mix --seed 42 --num_workers 0 --output_dir "knowledge_distilation/kd_results/${C12Method}_l020_c12_seed42$Suffix" --skip-test-evaluation
    } else {
      $LossMode = switch ($C12Method) { "adaface" { "adaface" }; "arcface" { "arcface" }; "subcenter" { "subcenter_arcface" } }
      & py -3.14 Eksperimen_Hardware_Aware_PDARTS/src/nas/retrain.py --genotype $Genotype --data_dir $Data --split_path $Split --output_dir "nas_results/retrain_l020_c12_${C12Method}_300e$Suffix/seed_42" --C_init 12 --num_cells 8 --stem_downsample 8 --reduction_indices "2,5" --epochs $Epochs --batch_size 64 --lr 0.001 --lr_min 0.000001 --weight_decay 0.05 --warmup_epochs $Warmup --drop_path_prob 0 --cutout_length 0 --augmentation_policy v4_robust_light --train_sampler pk --pk_p 16 --pk_k 4 --initial_weights $Initial12 --checkpoint_epochs $CheckpointEpoch --no_auxiliary --loss-mode $LossMode --label-smoothing 0 --arcface-margin 0.5 --arcface-scale 64 --arcface-margin-warmup-epochs 20 --subcenter-init-epsilon 0.001 --seed 42 --num_workers 0 --skip-test-evaluation
    }
  } else {
    $LossMode = switch ($Mode) { "ce_ls0" { "ce" }; "adaface" { "adaface" }; "arcface" { "arcface" }; "subcenter" { "subcenter_arcface" } }
    & py -3.14 Eksperimen_Hardware_Aware_PDARTS/src/nas/retrain.py --genotype $Genotype --data_dir $Data --split_path $Split --output_dir "nas_results/retrain_l020_c10_${Mode}_stabilized_300e$Suffix/seed_42" --C_init 10 --num_cells 8 --stem_downsample 8 --reduction_indices "2,5" --epochs $Epochs --batch_size 64 --lr 0.001 --lr_min 0.000001 --weight_decay 0.05 --warmup_epochs $Warmup --drop_path_prob 0 --cutout_length 0 --augmentation_policy v4_robust_light --train_sampler pk --pk_p 16 --pk_k 4 --initial_weights $Initial --checkpoint_epochs $CheckpointEpoch --no_auxiliary --loss-mode $LossMode --label-smoothing 0 --arcface-margin 0.5 --arcface-scale 64 --arcface-margin-warmup-epochs 20 --subcenter-init-epsilon 0.001 --seed 42 --num_workers 0 --skip-test-evaluation
  }
  if ($LASTEXITCODE -ne 0) { throw "Experiment '$Mode' failed with exit code $LASTEXITCODE" }
  if (-not $Smoke) { & py -3.14 scripts/c10_screening_ledger.py --run-id $RunId --branch $Gate --seed 42 --status complete }
} catch {
  if (-not $Smoke) { & py -3.14 scripts/c10_screening_ledger.py --run-id $RunId --branch $Gate --seed 42 --status failed }
  throw
}

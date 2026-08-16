param(
  [Parameter(Mandatory=$true)]
  [ValidateSet("audit","fcd","icd_full","select","confirm","final_eval","export")]
  [string]$Mode,
  [switch]$Smoke,
  [int[]]$Seeds = @(42),
  [switch]$AcknowledgeObservedTest
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$TeacherConfig = "nas_results/teacher_l020_c20_cells12_stem8_arcface_300e/seed_42/config.json"
$TeacherWeights = "nas_results/teacher_l020_c20_cells12_stem8_arcface_300e/seed_42/best_screening.pth"
$StudentConfig = "nas_results/retrain_l020_c10_arcface_stabilized_300e/seed_42/config.json"
$StudentBaseline = "nas_results/retrain_l020_c10_arcface_stabilized_300e/seed_42"
$Data = "preprocessed_results"
$Split = "PalmVein_Lightweight_Benchmark/dataset/splits/split_info.json"
$Calibration = "PalmVein_Lightweight_Benchmark/dataset/calibration_manifest.json"
$AuditDir = "results/diagnostics/c10_icd_teacher_audit_seed42"
$AuditResult = Join-Path $AuditDir "teacher_audit.json"
$SelectionDir = "results/diagnostics/c10_icd_screening_seed42"
$Selection = Join-Path $SelectionDir "selection.json"

function Assert-File([string]$Path) {
  if (-not (Test-Path $Path -PathType Leaf)) { throw "Required file not found: $Path" }
}

foreach ($Path in @($TeacherConfig,$TeacherWeights,$StudentConfig,(Join-Path $StudentBaseline "best_screening.pth"),$Split,$Calibration)) {
  Assert-File $Path
}
if (-not (Test-Path $Data -PathType Container)) { throw "Dataset directory not found: $Data" }

function Get-Initial([int]$Seed) {
  $Initial = "nas_results/controlled_initial_states/l020_c10_stem8_cells8_seed$Seed.pth"
  & py -3.14 scripts/create_c10_initial_state.py --config $StudentConfig --output $Initial --seed $Seed | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "Failed to create/reuse controlled initial state for seed $Seed" }
  return $Initial
}

function Get-RunDir([string]$Method, [int]$Seed, [bool]$IsSmoke=$false) {
  $Suffix = if ($IsSmoke) { "_smoke" } else { "" }
  if ($Method -eq "fcd") {
    return "knowledge_distilation/kd_results/fcd_l020_c10_c20ta_seed$Seed$Suffix"
  }
  return "knowledge_distilation/kd_results/icd_l020_c10_c20ta_seed$Seed$Suffix"
}

function Assert-AuditPass {
  Assert-File $AuditResult
  $Audit = Get-Content $AuditResult -Raw | ConvertFrom-Json
  if ($Audit.status -ne "PASS") {
    throw "Teacher geometry audit status is '$($Audit.status)'; ICD training is stopped."
  }
  if ($Audit.test_loader_created -ne $false -or $Audit.test_partition_inspected -ne $false) {
    throw "Teacher audit provenance indicates test access; refusing ICD screening."
  }
}

function Start-IcdRun([string]$Method, [int]$Seed, [bool]$IsSmoke=$false) {
  Assert-AuditPass
  $Initial = Get-Initial $Seed
  $Epochs = if ($IsSmoke) { 1 } else { 300 }
  $Warmup = if ($IsSmoke) { 1 } else { 10 }
  $SdcStart = if ($IsSmoke) { 1 } else { 76 }
  $IcdMode = if ($Method -eq "fcd") { "fcd" } else { "full" }
  $Output = Get-RunDir $Method $Seed $IsSmoke
  & py -3.14 knowledge_distilation/kd_train.py `
    --teacher_arch nas_eval `
    --teacher_config $TeacherConfig `
    --teacher_weights $TeacherWeights `
    --student_config $StudentConfig `
    --student_weights $Initial `
    --initial_student_weights $Initial `
    --no_pretrained_student `
    --data_dir $Data `
    --split_path $Split `
    --kd_method icd_compactness `
    --icd_mode $IcdMode `
    --icd_bank_size 5 `
    --icd_valid_steps 200 `
    --icd_delta 0.001 `
    --icd_gamma 50 `
    --icd_sdc_start_epoch $SdcStart `
    --icd_sdc_weight 0.5 `
    --icd_classification_weight 0.1 `
    --epochs $Epochs `
    --batch_size 64 `
    --lr 0.001 `
    --lr_min 0.000001 `
    --weight_decay 0.05 `
    --warmup_epochs $Warmup `
    --augmentation_policy v4_robust_light `
    --train_sampler pk `
    --pk_p 16 `
    --pk_k 4 `
    --label_smoothing 0 `
    --drop_path 0 `
    --cutout_length 0 `
    --no_mix `
    --seed $Seed `
    --num_workers 0 `
    --output_dir $Output `
    --skip-test-evaluation
  if ($LASTEXITCODE -ne 0) { throw "ICD run failed: method=$Method seed=$Seed" }
}

function Get-SelectedMethod {
  Assert-File $Selection
  $Payload = Get-Content $Selection -Raw | ConvertFrom-Json
  if ($Payload.status -ne "method_selected") {
    throw "Selection status is '$($Payload.status)'; no ICD method may proceed to test/export."
  }
  return [string]$Payload.winner.name
}

if ($Mode -eq "audit") {
  & py -3.14 knowledge_distilation/audit_intra_class_teacher.py `
    --teacher-config $TeacherConfig `
    --teacher-checkpoint $TeacherWeights `
    --student-config $StudentConfig `
    --student-checkpoint (Join-Path $StudentBaseline "best_screening.pth") `
    --data-dir $Data `
    --split-path $Split `
    --output-dir $AuditDir `
    --batch-size 64 `
    --num-workers 0
  exit $LASTEXITCODE
}

if ($Mode -in @("fcd","icd_full")) {
  Start-IcdRun $Mode 42 ([bool]$Smoke)
  exit 0
}

if ($Mode -eq "select") {
  & py -3.14 scripts/select_c10_icd.py `
    --baseline $StudentBaseline `
    --fcd (Get-RunDir "fcd" 42 $false) `
    --full (Get-RunDir "icd_full" 42 $false) `
    --output $Selection
  exit $LASTEXITCODE
}

$Selected = Get-SelectedMethod
if ($Mode -eq "confirm") {
  foreach ($Seed in $Seeds) {
    if ($Seed -eq 42) { continue }
    Start-IcdRun $Selected $Seed $false
  }
  exit 0
}

if ($Mode -in @("final_eval","export") -and -not $AcknowledgeObservedTest) {
  throw "$Mode requires -AcknowledgeObservedTest because this test split was previously observed."
}

foreach ($Seed in $Seeds) {
  $RunDir = Get-RunDir $Selected $Seed $false
  Assert-File (Join-Path $RunDir "best_screening.pth")
  $FinalDir = "results/final/c10_icd/seed_$Seed"
  New-Item -ItemType Directory -Force -Path $FinalDir | Out-Null
  if ($Mode -eq "final_eval") {
    & py -3.14 scripts/evaluate_frozen_identification.py `
      --config (Join-Path $RunDir "config.json") `
      --checkpoint (Join-Path $RunDir "best_screening.pth") `
      --data-dir $Data `
      --split-path $Split `
      --partition test `
      --acknowledge-observed-test `
      --output-dir (Join-Path $FinalDir "pytorch_test") `
      --batch-size 64 `
      --num-workers 0
    if ($LASTEXITCODE -ne 0) { throw "Final PyTorch evaluation failed for seed $Seed" }
  } else {
    & py -3.14 Eksperimen_Hardware_Aware_PDARTS/src/deployment/export_kd_onnx_int8.py `
      --model-dir $RunDir `
      --weights best_screening.pth `
      --output-stem model_benchmark `
      --calib-dir $Data `
      --num-calib 200 `
      --calibration-manifest $Calibration `
      --threads 4 `
      --warmup 20 `
      --runs 100 `
      --eval-accuracy `
      --acknowledge-observed-test `
      --data-dir $Data `
      --split-path $Split
    if ($LASTEXITCODE -ne 0) { throw "ONNX/PTQ export failed for seed $Seed" }
    Copy-Item (Join-Path $RunDir "model_benchmark.onnx") (Join-Path $FinalDir "model_benchmark_fp32.onnx") -Force
    Copy-Item (Join-Path $RunDir "model_benchmark_int8_static.onnx") (Join-Path $FinalDir "model_benchmark_int8_static.onnx") -Force
    foreach ($Name in @("config.json","model_benchmark_metadata.json","benchmark_int8_static_results.json","model_benchmark_acc.json","model_benchmark_int8_static_acc.json","model_benchmark_calibration_manifest_used.json")) {
      $Source = Join-Path $RunDir $Name
      if (Test-Path $Source) { Copy-Item $Source (Join-Path $FinalDir $Name) -Force }
    }
    Copy-Item $Selection (Join-Path $FinalDir "selection.json") -Force
  }
}

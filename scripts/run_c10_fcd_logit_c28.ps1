param(
  [Parameter(Mandatory=$true)]
  [ValidateSet("smoke","train","select","final_eval","export")]
  [string]$Mode,
  [switch]$AcknowledgeObservedTest
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$TeacherConfig = "nas_results/teacher_l020_c28_cells12_stem8_arcface_300e/seed_42/config.json"
$TeacherWeights = "nas_results/teacher_l020_c28_cells12_stem8_arcface_300e/seed_42/best_screening.pth"
$StudentConfig = "nas_results/retrain_l020_c10_arcface_stabilized_300e/seed_42/config.json"
$Initial = "nas_results/controlled_initial_states/l020_c10_stem8_cells8_seed42.pth"
$Baseline = "nas_results/retrain_l020_c10_arcface_stabilized_300e/seed_42"
$FcdControl = "knowledge_distilation/kd_results/fcd_l020_c10_c28ta_seed42"
$RunDir = "knowledge_distilation/kd_results/fcd_logit_t20_w005_l020_c10_c28ta_seed42"
$SmokeDir = "${RunDir}_smoke"
$Data = "preprocessed_results"
$Split = "PalmVein_Lightweight_Benchmark/dataset/splits/split_info.json"
$Calibration = "PalmVein_Lightweight_Benchmark/dataset/calibration_manifest.json"
$Audit = "results/diagnostics/c10_icd_c28_teacher_audit_seed42/teacher_audit.json"
$Selection = "results/diagnostics/c10_fcd_logit_c28_seed42/selection.json"
$FinalDir = "results/final/c10_fcd_logit_c28/seed_42"

function Assert-File([string]$Path) {
  if (-not (Test-Path $Path -PathType Leaf)) { throw "Required file not found: $Path" }
}

function Assert-Inputs {
  foreach ($Path in @(
    $TeacherConfig, $TeacherWeights, $StudentConfig, $Initial,
    (Join-Path $Baseline "screening_results.json"),
    (Join-Path $FcdControl "screening_results.json"),
    $Split, $Calibration, $Audit
  )) { Assert-File $Path }
  if (-not (Test-Path $Data -PathType Container)) {
    throw "Dataset directory not found: $Data"
  }
  $AuditPayload = Get-Content $Audit -Raw | ConvertFrom-Json
  if ($AuditPayload.status -ne "PASS") {
    throw "C28 teacher audit status is '$($AuditPayload.status)', expected PASS"
  }
  if ($AuditPayload.test_loader_created -ne $false -or $AuditPayload.test_partition_inspected -ne $false) {
    throw "C28 teacher audit indicates test access; refusing screening"
  }
}

function Start-FcdLogit([bool]$Smoke) {
  Assert-Inputs
  $Epochs = if ($Smoke) { 1 } else { 300 }
  $LrWarmup = if ($Smoke) { 1 } else { 10 }
  $LogitWarmup = if ($Smoke) { 1 } else { 20 }
  $SdcStart = if ($Smoke) { 1 } else { 76 }
  $Workers = if ($Smoke) { 0 } else { 4 }
  $Output = if ($Smoke) { $SmokeDir } else { $RunDir }
  & py -3.11 knowledge_distilation/kd_train.py `
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
    --icd_mode fcd `
    --icd_bank_size 5 `
    --icd_valid_steps 200 `
    --icd_delta 0.001 `
    --icd_gamma 50 `
    --icd_sdc_start_epoch $SdcStart `
    --icd_sdc_weight 0.5 `
    --icd_classification_weight 0.1 `
    --temperature 20 `
    --logit_kd_weight 0.05 `
    --icd_logit_warmup_epochs $LogitWarmup `
    --epochs $Epochs `
    --batch_size 64 `
    --lr 0.001 `
    --lr_min 0.000001 `
    --weight_decay 0.05 `
    --warmup_epochs $LrWarmup `
    --augmentation_policy v4_robust_light `
    --train_sampler pk `
    --pk_p 16 `
    --pk_k 4 `
    --label_smoothing 0 `
    --drop_path 0 `
    --cutout_length 0 `
    --no_mix `
    --seed 42 `
    --num_workers $Workers `
    --output_dir $Output `
    --skip-test-evaluation
  if ($LASTEXITCODE -ne 0) { throw "FCD+logit training failed with exit code $LASTEXITCODE" }
}

function Get-SelectedRun {
  Assert-File $Selection
  $Payload = Get-Content $Selection -Raw | ConvertFrom-Json
  if ($Payload.status -ne "method_selected" -or $Payload.winner.name -ne "fcd_logit_c28") {
    throw "FCD+logit was not selected on validation; final evaluation/export is blocked"
  }
  Assert-File (Join-Path $RunDir "best_screening.pth")
}

if ($Mode -eq "smoke") {
  Start-FcdLogit $true
  exit 0
}

if ($Mode -eq "train") {
  Start-FcdLogit $false
  exit 0
}

if ($Mode -eq "select") {
  Assert-Inputs
  & py -3.11 scripts/select_c10_fcd_logit_c28.py `
    --baseline $Baseline `
    --fcd $FcdControl `
    --fcd-logit $RunDir `
    --output $Selection
  exit $LASTEXITCODE
}

if (-not $AcknowledgeObservedTest) {
  throw "$Mode requires -AcknowledgeObservedTest because this test split was previously observed"
}
Get-SelectedRun
New-Item -ItemType Directory -Force -Path $FinalDir | Out-Null

if ($Mode -eq "final_eval") {
  & py -3.11 scripts/evaluate_frozen_identification.py `
    --config (Join-Path $RunDir "config.json") `
    --checkpoint (Join-Path $RunDir "best_screening.pth") `
    --data-dir $Data `
    --split-path $Split `
    --partition test `
    --acknowledge-observed-test `
    --output-dir (Join-Path $FinalDir "pytorch_test") `
    --batch-size 64 `
    --num-workers 0
  if ($LASTEXITCODE -ne 0) { throw "Frozen test evaluation failed" }
  exit 0
}

& py -3.11 Eksperimen_Hardware_Aware_PDARTS/src/deployment/export_kd_onnx_int8.py `
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
if ($LASTEXITCODE -ne 0) { throw "ONNX/PTQ export failed" }

Copy-Item (Join-Path $RunDir "model_benchmark.onnx") (Join-Path $FinalDir "model_benchmark_fp32.onnx") -Force
Copy-Item (Join-Path $RunDir "model_benchmark_int8_static.onnx") (Join-Path $FinalDir "model_benchmark_int8_static.onnx") -Force
foreach ($Name in @(
  "config.json", "screening_results.json", "model_benchmark_metadata.json",
  "benchmark_int8_static_results.json", "model_benchmark_acc.json",
  "model_benchmark_int8_static_acc.json",
  "model_benchmark_calibration_manifest_used.json"
)) {
  $Source = Join-Path $RunDir $Name
  if (Test-Path $Source) { Copy-Item $Source (Join-Path $FinalDir $Name) -Force }
}
Copy-Item $Selection (Join-Path $FinalDir "selection.json") -Force

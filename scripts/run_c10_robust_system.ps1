param(
  [Parameter(Mandatory=$true)]
  [ValidateSet("audit","smoke","norm","consistency","avgpool","select","confirm","final_eval","export")]
  [string]$Mode,
  [int[]]$Seeds = @(42),
  [switch]$AcknowledgeObservedTest
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Python = "py"
$PythonVersion = "-3.14"
$Retrain = "Eksperimen_Hardware_Aware_PDARTS/src/nas/retrain.py"
$Genotype = "nas_results/search_hwint8_l0.20/genotype_final.json"
$BaselineDir = "nas_results/retrain_l020_c10_arcface_stabilized_300e/seed_42"
$BaselineConfig = Join-Path $BaselineDir "config.json"
$Data = "preprocessed_results"
$Split = "PalmVein_Lightweight_Benchmark/dataset/splits/split_info.json"
$Calibration = "PalmVein_Lightweight_Benchmark/dataset/calibration_manifest.json"
$DiagnosticRoot = "results/diagnostics/c10_robust_system_seed42"
$Selection = Join-Path $DiagnosticRoot "selection.json"

function Assert-File([string]$Path) {
  if (-not (Test-Path $Path -PathType Leaf)) { throw "Required file not found: $Path" }
}

foreach ($Path in @($Retrain,$Genotype,$BaselineConfig,(Join-Path $BaselineDir "best_screening.pth"),$Split)) {
  Assert-File $Path
}
if (-not (Test-Path $Data -PathType Container)) { throw "Dataset directory not found: $Data" }

function Get-Initial([int]$Seed) {
  $Initial = "nas_results/controlled_initial_states/l020_c10_stem8_cells8_seed$Seed.pth"
  & $Python $PythonVersion scripts/create_c10_initial_state.py `
    --config $BaselineConfig --output $Initial --seed $Seed --C-init 10 --num-cells 8 --stem-downsample 8 | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "Failed to create/reuse controlled initial state for seed $Seed" }
  return $Initial
}

function Get-RunDir([string]$Recipe, [int]$Seed, [bool]$Smoke=$false) {
  $Suffix = if ($Smoke) { "_smoke" } else { "" }
  if ($Recipe -eq "norm") { return "nas_results/retrain_l020_c10_robust_norm_300e$Suffix/seed_$Seed" }
  if ($Recipe -eq "consistency") { return "nas_results/retrain_l020_c10_robust_consistency_300e$Suffix/seed_$Seed" }
  if ($Recipe -eq "avgpool") { return "nas_results/retrain_l020_c10_robust_avgpool_300e$Suffix/seed_$Seed" }
  throw "Unknown recipe: $Recipe"
}

function Get-RobustnessResult([string]$Recipe, [int]$Seed=42) {
  if ($Recipe -eq "baseline") { return Join-Path $DiagnosticRoot "baseline_robustness.json" }
  return Join-Path $DiagnosticRoot "$($Recipe)_seed$($Seed)_robustness.json"
}

function Invoke-Robustness([string]$Recipe, [string]$RunDir, [int]$Seed=42) {
  $Output = Get-RobustnessResult $Recipe $Seed
  & $Python $PythonVersion scripts/evaluate_c10_robustness.py `
    --run-dir $RunDir `
    --data-dir $Data `
    --split-path $Split `
    --output $Output `
    --batch-size 64 `
    --num-workers 0
  if ($LASTEXITCODE -ne 0) { throw "Robustness suite failed: $Recipe seed=$Seed" }
}

function Invoke-Training([string]$Recipe, [int]$Seed, [bool]$Smoke=$false) {
  $Initial = Get-Initial $Seed
  $Output = Get-RunDir $Recipe $Seed $Smoke
  $Epochs = if ($Smoke) { 1 } else { 300 }
  $Warmup = if ($Smoke) { 1 } else { 10 }
  $StemPool = if ($Recipe -eq "avgpool") { "avg" } else { "max" }
  $ConsistencyMode = if ($Recipe -in @("consistency","avgpool")) { "js_two_view" } else { "none" }

  & $Python $PythonVersion $Retrain `
    --genotype $Genotype `
    --data_dir $Data `
    --split_path $Split `
    --output_dir $Output `
    --C_init 10 `
    --num_cells 8 `
    --stem_downsample 8 `
    --stem-pool $StemPool `
    --reduction_indices "2,5" `
    --epochs $Epochs `
    --batch_size 64 `
    --lr 0.001 `
    --lr_min 0.000001 `
    --weight_decay 0.05 `
    --warmup_epochs $Warmup `
    --drop_path_prob 0 `
    --cutout_length 0 `
    --augmentation_policy v4_robust_light `
    --input-profile robust_percentile_v1 `
    --consistency-mode $ConsistencyMode `
    --consistency-temperature 4 `
    --consistency-weight 1.0 `
    --consistency-ramp-epochs 20 `
    --train_sampler pk `
    --pk_p 16 `
    --pk_k 4 `
    --no_auxiliary `
    --loss-mode arcface `
    --label-smoothing 0 `
    --arcface-margin 0.5 `
    --arcface-scale 64 `
    --arcface-margin-warmup-epochs 20 `
    --initial_weights $Initial `
    --seed $Seed `
    --num_workers 0 `
    --skip-test-evaluation
  if ($LASTEXITCODE -ne 0) { throw "Training failed: $Recipe seed=$Seed" }

  $Screening = Join-Path $Output "screening_results.json"
  Assert-File $Screening
  $Payload = Get-Content $Screening -Raw | ConvertFrom-Json
  if ($Payload.status -ne "screening_complete_test_not_evaluated") {
    throw "Screening did not record validation-only status: $Screening"
  }
  if ($Payload.test_loader_created -ne $false -or $Payload.test_partition_inspected -ne $false) {
    throw "Screening provenance indicates test access: $Screening"
  }
  if (-not $Smoke) { Invoke-Robustness $Recipe $Output $Seed }
}

function Get-Selection {
  Assert-File $Selection
  $Payload = Get-Content $Selection -Raw | ConvertFrom-Json
  if ($Payload.status -ne "method_selected") {
    throw "No new robust system passed the frozen acceptance rule; retain E0."
  }
  return $Payload
}

if ($Mode -eq "audit") {
  Get-Initial 42 | Out-Null
  Invoke-Robustness "baseline" $BaselineDir 42
  exit 0
}

if ($Mode -eq "smoke") {
  foreach ($Recipe in @("norm","consistency","avgpool")) {
    Invoke-Training $Recipe 42 $true
  }
  exit 0
}

if ($Mode -in @("norm","consistency","avgpool")) {
  Invoke-Training $Mode 42 $false
  exit 0
}

if ($Mode -eq "select") {
  & $Python $PythonVersion scripts/select_c10_robust_system.py `
    --baseline (Get-RobustnessResult "baseline") `
    --norm (Get-RobustnessResult "norm") `
    --consistency (Get-RobustnessResult "consistency") `
    --avgpool (Get-RobustnessResult "avgpool") `
    --output $Selection
  exit $LASTEXITCODE
}

$Selected = Get-Selection
$Recipe = [string]$Selected.winner.name
if ($Recipe -eq "robust_norm") { $Recipe = "norm" }
elseif ($Recipe -eq "robust_norm_js") { $Recipe = "consistency" }
elseif ($Recipe -eq "robust_norm_avgpool_js") { $Recipe = "avgpool" }
else { throw "Unexpected selected recipe: $Recipe" }

if ($Mode -eq "confirm") {
  foreach ($Seed in $Seeds) {
    if ($Seed -eq 42) { continue }
    Invoke-Training $Recipe $Seed $false
  }
  exit 0
}

if ($Mode -eq "final_eval" -and -not $AcknowledgeObservedTest) {
  throw "final_eval requires -AcknowledgeObservedTest because this test split was previously observed."
}

foreach ($Seed in $Seeds) {
  $RunDir = Get-RunDir $Recipe $Seed $false
  $RobustnessPath = Get-RobustnessResult $Recipe $Seed
  Assert-File $RobustnessPath
  $Robustness = Get-Content $RobustnessPath -Raw | ConvertFrom-Json
  $Checkpoint = [string]$Robustness.winner.checkpoint
  $Config = Join-Path $RunDir "config.json"
  Assert-File $Checkpoint
  Assert-File $Config
  $FinalDir = "results/final/c10_robust_system/seed_$Seed"
  New-Item -ItemType Directory -Force -Path $FinalDir | Out-Null

  if ($Mode -eq "final_eval") {
    & $Python $PythonVersion scripts/evaluate_frozen_identification.py `
      --config $Config `
      --checkpoint $Checkpoint `
      --data-dir $Data `
      --split-path $Split `
      --partition test `
      --acknowledge-observed-test `
      --output-dir (Join-Path $FinalDir "pytorch_test") `
      --batch-size 64 `
      --num-workers 0
    if ($LASTEXITCODE -ne 0) { throw "Final test evaluation failed for seed $Seed" }
  }
  elseif ($Mode -eq "export") {
    Assert-File $Calibration
    & $Python $PythonVersion Eksperimen_Hardware_Aware_PDARTS/src/deployment/export_kd_onnx_int8.py `
      --model-dir $RunDir `
      --weights (Split-Path $Checkpoint -Leaf) `
      --output-stem model_benchmark `
      --calib-dir $Data `
      --num-calib 200 `
      --calibration-manifest $Calibration `
      --threads 4 `
      --warmup 20 `
      --runs 100 `
      --data-dir $Data `
      --split-path $Split
    if ($LASTEXITCODE -ne 0) { throw "ONNX/PTQ export failed for seed $Seed" }
    Copy-Item (Join-Path $RunDir "model_benchmark.onnx") (Join-Path $FinalDir "model_benchmark_fp32.onnx") -Force
    Copy-Item (Join-Path $RunDir "model_benchmark_int8_static.onnx") (Join-Path $FinalDir "model_benchmark_int8_static.onnx") -Force
    foreach ($Name in @("config.json","model_benchmark_metadata.json","benchmark_int8_static_results.json","model_benchmark_calibration_manifest_used.json")) {
      $Source = Join-Path $RunDir $Name
      if (Test-Path $Source) { Copy-Item $Source (Join-Path $FinalDir $Name) -Force }
    }
    Copy-Item $Selection (Join-Path $FinalDir "selection.json") -Force
    Copy-Item $RobustnessPath (Join-Path $FinalDir "validation_robustness.json") -Force
  }
}

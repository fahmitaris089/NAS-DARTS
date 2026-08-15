# C10 bounded validation-screening workflow

This workflow targets closed-set identification on the existing SCUT image-level
80:10:10 split. It does not establish subject- or session-disjoint generalization.
The test partition has already been observed in earlier experiments and is never
used by the commands below for method selection.

## 1. Mandatory forensics

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_c10_targeted_screening.ps1 -Mode forensics
```

The command creates `results/diagnostics/c10_error_forensics_seed42/`. It loads
the four frozen C10 validation checkpoints, verifies their recorded screening
epochs, constructs class prototypes from training images only, and writes the
decision to `decision_gate.json`. No test loader is constructed.

## 2. Run the selected two-run branch

Use only the branch named by `recommended_branch`:

```powershell
# teacher correct on common student errors
powershell -ExecutionPolicy Bypass -File scripts\run_c10_targeted_screening.ps1 -Mode progressive_center
powershell -ExecutionPolicy Bypass -File scripts\run_c10_targeted_screening.ps1 -Mode progressive_hybrid

# feature-norm finding
powershell -ExecutionPolicy Bypass -File scripts\run_c10_targeted_screening.ps1 -Mode ce_ls0
powershell -ExecutionPolicy Bypass -File scripts\run_c10_targeted_screening.ps1 -Mode adaface

# multi-mode/prototype finding
powershell -ExecutionPolicy Bypass -File scripts\run_c10_targeted_screening.ps1 -Mode arcface
powershell -ExecutionPolicy Bypass -File scripts\run_c10_targeted_screening.ps1 -Mode subcenter
```

`-ForceBranch` exists only for a documented protocol amendment. Normal use is
gate-controlled. Non-smoke seed-42 runs are registered in
`results/diagnostics/c10_screening_ledger.json`; a fifth run is rejected.

If the decision is complementarity, do not start new training:

```powershell
py -3.14 knowledge_distilation/validation_ensemble_soup.py
```

Average-logit ensembling is diagnostic. Weight soup is attempted only if that
ensemble reaches 834/834 validation. BatchNorm is recalibrated with training
images only.

## 3. Select or use the remaining two runs

Rank results with their `screening_results.json` files:

```powershell
py -3.14 scripts/select_c10_screening.py PATH1\screening_results.json PATH2\screening_results.json
```

The locked rule is validation errors, ordinary CE loss, then mean true-class
margin. If neither branch run beats the locked PK-CE baseline, run C12 PK-CE:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_c10_targeted_screening.ps1 -Mode c12_pk_ce
py -3.14 scripts/compare_c12_gate.py
```

Only if `allow_method_run=true`, apply the best branch method as the fourth run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_c10_targeted_screening.ps1 `
  -Mode c12_branch -C12Method progressive_hybrid
```

Do not add cells or start a fifth seed-42 screening run.

## 4. Confirmation and final evaluation

After the method is frozen, rerun the exact configuration for seeds 123 and
2026. Evaluate one frozen checkpoint per seed. A test run requires explicit
acknowledgement of prior test exposure:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_frozen_c10_confirmation.ps1 `
  -Method progressive_hybrid -CInit 10 -ConfirmMethodFrozen

py -3.14 scripts/evaluate_frozen_identification.py `
  --config PATH\config.json `
  --checkpoint PATH\best_screening.pth `
  --data-dir preprocessed_results `
  --split-path PalmVein_Lightweight_Benchmark/dataset/splits/split_info.json `
  --partition test `
  --acknowledge-observed-test `
  --output-dir results/final/METHOD/seed_42
```

The final evaluator reports CRR/accuracy and correct/total only. It deliberately
does not calculate EER, FAR, FRR, or biometric AUC. Paired errors can be compared
with:

```powershell
py -3.14 scripts/paired_mcnemar.py BASELINE\predictions.csv CANDIDATE\predictions.csv
```

ONNX fusion, FP32 parity, training-only PTQ calibration, INT8 accuracy, and the
fixed Raspberry Pi benchmark follow only after all three optimization seeds are
complete. A 100% run is an engineering target, not a guaranteed or required
scientific conclusion.

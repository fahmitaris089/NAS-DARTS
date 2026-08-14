param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("dkd", "adaface", "adaface_dkd")]
    [string]$Mode,
    [string]$PythonVersion = "3.14"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Teacher = "Eksperimen_Hardware_Aware_PDARTS/checkpoints/teacher/EfficientNetV2M_best_model.pth"
$Data = "preprocessed_results"
$Split = "PalmVein_Lightweight_Benchmark/dataset/splits/split_info.json"
$CEConfig = "nas_results/retrain_hwint8_l020_c8_stem8_robust_300e/seed_42/config.json"
$CEWeights = "nas_results/retrain_hwint8_l020_c8_stem8_robust_300e/seed_42/best_model.pth"
$AdaDir = "nas_results/retrain_hwint8_l020_c8_stem8_adaface_300e/seed_42"

if ($Mode -eq "dkd") {
    & py "-$PythonVersion" knowledge_distilation/kd_train.py `
        --teacher_arch efficientnet_v2_m --teacher_weights $Teacher `
        --student_config $CEConfig --student_weights $CEWeights `
        --data_dir $Data --split_path $Split `
        --kd_method dkd --temperature 4 --dkd_alpha 1 --dkd_beta 8 --dkd_warmup_epochs 20 `
        --epochs 75 --batch_size 64 --lr 0.00003 --lr_min 0.000001 --weight_decay 0.02 `
        --warmup_epochs 5 --scheduler cosine --augmentation_policy v4_robust_light `
        --cutout_length 0 --drop_path 0 --label_smoothing 0 --no_mix `
        --seed 42 --num_workers 0 --skip-test-evaluation `
        --output_dir knowledge_distilation/kd_results/dkd_l020_c8_stem8_robust_seed42
}
elseif ($Mode -eq "adaface") {
    & py "-$PythonVersion" Eksperimen_Hardware_Aware_PDARTS/src/nas/retrain.py `
        --genotype nas_results/search_hwint8_l0.20/genotype_final.json `
        --data_dir $Data --split_path $Split --output_dir $AdaDir `
        --C_init 8 --num_cells 8 --stem_downsample 8 --reduction_indices 2,5 `
        --epochs 300 --batch_size 64 --lr 0.001 --lr_min 0.000001 `
        --weight_decay 0.05 --warmup_epochs 10 `
        --drop_path_prob 0 --cutout_length 0 --augmentation_policy v4_robust_light `
        --loss-mode adaface --adaface-m 0.4 --adaface-h 0.333 --adaface-s 64 `
        --adaface-t-alpha 0.01 --no_auxiliary --seed 42 --num_workers 0 `
        --skip-test-evaluation
}
else {
    if (-not (Test-Path "$AdaDir/best_model.pth")) {
        throw "AdaFace checkpoint is missing. Run -Mode adaface and pass the validation gate first."
    }
    & py "-$PythonVersion" knowledge_distilation/kd_train.py `
        --teacher_arch efficientnet_v2_m --teacher_weights $Teacher `
        --student_config "$AdaDir/config.json" --student_weights "$AdaDir/best_model.pth" `
        --data_dir $Data --split_path $Split --adaface `
        --kd_method dkd --temperature 4 --dkd_alpha 1 --dkd_beta 8 --dkd_warmup_epochs 20 `
        --epochs 75 --batch_size 64 --lr 0.00003 --lr_min 0.000001 --weight_decay 0.02 `
        --warmup_epochs 5 --scheduler cosine --augmentation_policy v4_robust_light `
        --cutout_length 0 --drop_path 0 --label_smoothing 0 --no_mix `
        --seed 42 --num_workers 0 --skip-test-evaluation `
        --output_dir knowledge_distilation/kd_results/adaface_dkd_l020_c8_stem8_robust_seed42
}

if ($LASTEXITCODE -ne 0) {
    throw "Experiment '$Mode' failed with exit code $LASTEXITCODE"
}

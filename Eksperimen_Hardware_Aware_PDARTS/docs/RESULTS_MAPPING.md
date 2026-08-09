# Pemetaan Hasil Tesis

Pemetaan terstruktur tersedia pada `results/thesis_manifest.csv`.

- LUT operator: `results/lut/`
- Genotype lambda 0.00-0.20: `results/search/`
- Retraining awal dan refinement: `results/retraining/`
- Seluruh 14 eksperimen teacher: `results/teacher/`
- Perbandingan lintas teacher: `results/teacher/comparison/`
- Eksperimen KD C8, C10, dan C12: `results/kd/`
- Evaluasi ONNX FP32/INT8: `results/deployment/`
- Checkpoint student final sebelum dan sesudah KD: `checkpoints/student/`
- Model deployment final: `models/onnx_fp32/` dan `models/onnx_int8/`

Angka utama model final:

- FP32 setelah KD: 99.76%
- INT8: 99.64%
- ukuran INT8: 0.928 MB
- mean latency: 3.87 ms
- median latency: 3.76 ms
- p95 latency: 4.52 ms

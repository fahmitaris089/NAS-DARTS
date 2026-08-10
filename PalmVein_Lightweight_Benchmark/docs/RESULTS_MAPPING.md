# Results mapping

| Artefak | Sumber | Makna |
|---|---|---|
| `results/model_validation/model_spec_validation.csv` | `scripts/validate_models.py` | output shape, parameter, MMAC, audit Ding, serta diagnostik PalmNet terhadap tabel paper |
| `results/scratch/<model>/seed_<seed>/run_config.json` | `scripts/train.py` | snapshot konfigurasi setiap run |
| `results/scratch/<model>/seed_<seed>/training_log.csv` | training loop | metrik train/validation per epoch |
| `artifacts/checkpoints/.../best.pth` | minimum validation loss | satu-satunya checkpoint untuk test/deployment |
| `results/.../test_results.json` | evaluasi setelah seleksi | akurasi FP32 PyTorch per seed |
| `results/deployment/*onnx_fp32.json` | `scripts/export_onnx.py` | hash, ukuran, dan parity ONNX |
| `results/deployment/*quantization.json` | `scripts/quantize_int8.py` | konfigurasi PTQ, validasi manifest, akurasi INT8 |
| `results/deployment/*latency.json` | `scripts/benchmark_raspberry_pi.py` | latency dan metadata platform |
| `results/summary/summary_scratch_fp32.csv` | `scripts/summarize_results.py` | mean/std scratch per model |
| `results/summary/summary_pretrained_fp32.csv` | `scripts/summarize_results.py` | mean/std pretrained terpisah |
| `results/summary/summary_int8.csv` | `scripts/summarize_results.py` | akurasi/ukuran INT8 per model-seed |
| `results/summary/raspberry_pi_latency.csv` | `scripts/summarize_results.py` | mean/median/p95 per model-seed |
| `results/migrations/mnasnet_a1_to_b1_torchvision.json` | `scripts/migrate_mnasnet_a1_legacy.py` | audit perubahan label hasil MnasNet lama dan bukti hash tensor tetap identik |
| `results/migrations/ding_pruned_to_legacy_v1.json` | `scripts/migrate_ding_pruned_legacy.py` | audit pemindahan hasil Ding lima blok lama dan bukti hash tensor tetap identik |

File ringkasan yang kosong berarti eksperimen sumber belum dijalankan; nilai kosong tidak boleh diubah menjadi nol atau `N/A` tanpa membedakan “belum dijalankan” dari “tidak berlaku”.

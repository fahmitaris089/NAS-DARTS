# Lingkungan Perangkat

## Training

Tahap search, retraining, teacher, dan KD dirancang untuk PyTorch dengan GPU
CUDA. Versi paket yang diperlukan dicantumkan pada
`requirements/training.txt`.

## Raspberry Pi 5

LUT dan benchmark tesis menggunakan:

- Raspberry Pi 5
- ONNX Runtime CPU
- 4 thread
- warm-up sebelum pencatatan latency
- model ONNX FP32 dan ONNX INT8

LUT bersifat spesifik terhadap perangkat dan lingkungan eksekusi. Jika
perangkat, runtime, presisi, atau jumlah thread berubah, probe operator harus
diukur ulang. Jangan menganggap nilai LUT tesis berlaku universal.


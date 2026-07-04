# METODE EKSPERIMEN AKTUAL
## Dokumentasi Lengkap Metode yang Digunakan dalam Penelitian

**Judul Thesis (Revisi):**  
*Arsitektur Jaringan Ringan untuk Pengenalan Palm Vein pada Perangkat Edge Menggunakan Hardware-Aware Neural Architecture Search dan Knowledge Distillation*

---

## 1. DATASET DAN PREPROCESSING

### 1.1 Dataset
- **Sumber**: SCUT_PV_v1 Palm Vein Dataset (internal)
- **Jumlah kelas**: 834 subjek
- **Total sampel**: ~50,000 citra NIR grayscale
- **Resolusi input**: 224×224 piksel
- **Format**: BMP grayscale → RGB 3-channel (replicate untuk kompatibilitas ImageNet)

### 1.2 Split Data
- **Training**: 70% per-class stratified
- **Validation**: 15% (untuk hyperparameter tuning dan early stopping)
- **Test**: 15% (held-out untuk evaluasi final)
- **Seed**: 42 (reproducibility)

### 1.3 Preprocessing Pipeline
1. **ROI Extraction**: Deteksi region telapak tangan menggunakan thresholding
2. **CLAHE**: Contrast Limited Adaptive Histogram Equalization untuk enhance pola vena
3. **Resize**: 224×224 piksel
4. **Grayscale to RGB**: Replikasi single channel ke 3 channel
5. **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 1.4 Data Augmentation (Training Only)
- Random rotation: ±10°
- Random translation: ±5%
- Random brightness/contrast: ±10%
- CutOut: 16×16 patch masking
- Drop path: Linear scheduling 0 → 0.2

---

## 2. HARDWARE-AWARE NEURAL ARCHITECTURE SEARCH

### 2.1 Search Space
**Operator Primitives** (12 operators):
```python
PRIMITIVES = [
    'none',           # zero operation (edge pruning)
    'skip_connect',   # identity / factorized reduce
    'sep_conv_3x3',   # separable convolution 3×3
    'sep_conv_5x5',   # separable convolution 5×5
    'dil_conv_3x3',   # dilated convolution 3×3
    'dil_conv_5x5',   # dilated convolution 5×5
    'mbconv3_3x3',    # MobileNetV2 inverted residual (expand=3)
    'mbconv6_3x3',    # MobileNetV2 inverted residual (expand=6)
    'rep_conv_3x3',   # re-parameterizable conv 3×3 (RepVGG-style)
    'rep_conv_5x5',   # re-parameterizable conv 5×5
    'avg_pool_3x3',   # average pooling 3×3
    'max_pool_3x3',   # max pooling 3×3
]
```

**Cell-based Architecture**:
- Mengadopsi paradigma DARTS cell-based search
- Normal cells: feature transformation
- Reduction cells: spatial downsampling
- Setiap cell = DAG dengan 4 intermediate nodes
- Top-2 edges per node retained dalam genotype

### 2.2 Progressive DARTS (P-DARTS)
**Metode**: P-DARTS (Progressive Differentiable Architecture Search)

**3-Stage Progressive Search**:

| Stage | Cells | Epochs | Num Ops | Alpha Warmup |
|-------|-------|--------|---------|--------------|
| 1     | 5     | 25     | 12      | 10 epochs    |
| 2     | 8     | 25     | 7       | 10 epochs    |
| 3     | 11    | 25     | 4       | 10 epochs    |

**Key Features**:
- **Progressive depth**: Bridges search-eval depth gap
- **Operation pruning**: Prune weak operations between stages
- **Alpha warmup**: Weight-only training untuk N epochs pertama per stage
- **Skip-connect dropout**: Linear scheduling 0.0 → 0.5 (regularisasi)
- **Diversity guard**: Minimal 2 conv ops retained (prevent collapse)

### 2.3 Hardware-Aware Search (Latency Penalty)

**Objective Function**:
```
L_total = L_CE + λ × Latency_expected

where:
  Latency_expected = Σ_edges Σ_ops softmax(α_edge)[op] × LUT[op]
```

**Latency Lookup Table (LUT) Construction**:
1. **Fase Export** (Mac/GPU dengan PyTorch):
   - Bangun standalone ONNX untuk setiap operator
   - Konfigurasi: (C, H, stride) = {(8,56,1), (16,28,1), (32,14,1), (16,28,2), (32,14,2)}
   - Simpan manifest.json + ONNX files

2. **Fase Measure** (Raspberry Pi 5):
   - Profiling menggunakan ONNX Runtime CPU (4 threads)
   - 100 iterations per operator (20 warmup)
   - Median latency per config → mean across configs
   - Output: `latency_lut_pi.json`

**Hardware Target**: Raspberry Pi 5 (Cortex-A76, 4GB RAM, 4 threads)

**Lambda Values Explored**: λ ∈ {0.0, 0.05, 0.10, 0.20}
- λ=0.0: Pure DARTS (accuracy-only)
- λ=0.05: Balanced (best Pareto)
- λ=0.10/0.20: Latency-aggressive

### 2.4 Search Hyperparameters

**Supernet Configuration**:
- `C_search = 16` (initial channels)
- Input size: 112×112 (search phase, lebih kecil untuk efisiensi)
- Batch size: 16

**Weight Optimizer (SGD)**:
- Learning rate: 0.025 → 0.001 (cosine annealing)
- Momentum: 0.9
- Weight decay: 3e-4
- Gradient clipping: 5.0

**Architecture Optimizer (Adam)**:
- Learning rate: 6e-4
- Betas: (0.5, 0.999)
- Weight decay: 1e-3

**Bilevel Optimization**:
- First-order approximation (efficient DARTS)
- Alternating updates: α on val batch, w on train batch

### 2.5 Retrain Phase

Setelah search convergence, arsitektur terbaik di-retrain dari scratch:

**Architecture Sizing**:
- Auto-tuned `C_init` untuk target 250k-400k parameters
- `num_cells = 8`
- Auxiliary head: weight 0.4 (at 2/3 network depth)
- Stem downsample: 4× (spatial reduction 224→56)
- Reduction indices: [2, 5]

**Training Configuration**:
- Optimizer: AdamW (lr=1e-3, weight_decay=0.05)
- Scheduler: Warmup 10 epochs → Cosine annealing to 1e-6
- Epochs: 600
- Batch size: 64
- Label smoothing: 0.2
- Drop path: 0.2
- Dropout: 0.3
- Cutout: 16×16

---

## 3. KNOWLEDGE DISTILLATION

### 3.1 Teacher Model
**Architecture**: EfficientNet-V2-Medium  
**Performance**: 100% training accuracy pada SCUT_PV_v1 834 kelas  
**Parameters**: ~53.9M  
**Status**: Frozen (eval mode, no gradient updates)

**Teacher Training**:
- Pretrained ImageNet weights (transfer learning)
- Fine-tune 100 epochs
- Optimizer: AdamW (lr=3e-4, wd=0.01)
- Augmentation: RandAugment + Mixup α=0.2

**Alternative Teachers Benchmarked** (9 models):
- ResNet50: 100% acc, 25.2M params (tercepat train)
- ConvNeXtBase: 100% acc, 88.4M params
- RegNetY16GF: 100% acc, 83.1M params
- DenseNet121: 99.88% acc, 7.8M params
- MobileNetV3-Large: 99.88% acc, 5.3M params
- EfficientNetB4: 99.76% acc, 19.0M params
- InceptionV3: 99.76% acc, 26.7M params
- VGG16: 99.64% acc, 137.7M params

### 3.2 Student Model
**Architecture**: EvalNetwork (dari P-DARTS genotype)
- Genotype: hasil NAS (mobile_v2, rep_conv, atau hwNAS topology)
- `C_init`: 4, 6, atau 8 (varies by capacity)
- `num_cells = 8`
- Auxiliary head: **disabled** (auxiliary=False) untuk KD
- Pretrained: loaded from retrain phase (NOKD baseline)

### 3.3 Distillation Method
**Loss Function** (Hinton KD):
```python
L_total = α × L_CE(y_hard, student_logits) + 
          (1-α) × T² × KL(softmax(teacher_logits/T) || softmax(student_logits/T))

where:
  α = balance weight (CE vs KD)
  T = temperature (softmax smoothing)
```

**Hyperparameter Grid Search**:
- Temperature T: {4, 6, 8}
- Balance α: {0.1, 0.2, 0.3}
- Best config selected by validation accuracy

**Training Configuration**:
- Epochs: 150-500 (varies by experiment)
- Optimizer: AdamW
- Learning rate: 1e-3 → 1e-6 (cosine annealing with warmup)
- Warmup: 10 epochs
- Batch size: 64
- Label smoothing: 0.0 (disabled during KD, teacher provides soft labels)
- AMP: Enabled (mixed precision training)

**MixUp/CutMix** (Optional):
- MixUp alpha: 0.8 (Beta distribution)
- CutMix alpha: 1.0
- Mix probability: 1.0
- Switch probability: 0.5 (MixUp vs CutMix)
- Note: Some experiments run without mixing (nomix)

### 3.4 KD Results
**Observed Gains**:
- mobile_v2_C3: 96.04% (NOKD) → 97.00% (KD) = **+0.96%**
- mobile_v2_C4: 98.56% (NOKD) → 98.92% (KD) = **+0.36%**

**Insight**: KD gain inversely proportional to student capacity (larger gap when student is smaller)

**hwNAS models**: KD flat/marginal (~0.0-0.2% gain) → task separability tinggi, headroom terbatas

---

## 4. MODEL COMPRESSION DAN QUANTIZATION

### 4.1 Post-Training Quantization (PTQ)

**Method**: Static INT8 Quantization (per-channel)

**Quantization Recipe**:
```python
- Format: QDQ (QuantizeLinear/DequantizeLinear)
- Activation: QInt8
- Weight: QInt8
- Per-channel: True (MANDATORY, per-tensor degrades quality)
- Opset: ≥13 (required for per-channel)
- Calibration: 200 images from training set
```

**Pre-processing** (ORT best practice):
1. **Opset upgrade**: Ensure opset ≥13 (avoid silent per-channel disable)
2. **Symbolic shape inference**: `quant_pre_process()` untuk avoid degenerate bias scales

**Calibration**:
- Dataset: 200 training images (stratified sampling)
- Preprocessing: Same pipeline sebagai training (CLAHE + normalization)

### 4.2 Quantization Results

**Size Compression** (konsisten across models):
- hwNAS λ0.05 C6: 0.79 MB → 0.45 MB (1.76×)
- hwNAS λ0.20 C8: 1.53 MB → 0.61 MB (2.51×)
- repconv_C8: 1.46 MB → 0.60 MB (2.43×)

**Latency Trade-off** (architecture-dependent):
- **C4/C6 (compact cells)**: INT8 **slower** than FP32 (0.67-0.95×)
- **C8 (dense cells)**: INT8 **faster** than FP32 (1.06-1.19×)

**Root Cause** (validated via operator profiling):
- INT8 benefit = (compute saving) - (QDQ overhead)
- Compact/memory-bound cells: low arithmetic intensity → QDQ overhead dominates → net loss
- Dense/compute-bound cells: high arithmetic intensity → compute saving dominates → net gain
- Crossover point: empirically around C8

**Accuracy Impact**:
- Most models: ≤0.5% drop (acceptable)
- mobile_v2_C3: -1.08% drop (outlier, small capacity)
- hwNAS C4/C6/C8: 0.0-0.3% drop (robust)

### 4.3 Deployment Decision Rule
**Per-model basis**:
- **Compact cells (C4/C6) memory-bound**: Deploy FP32 for latency; INT8 for storage
- **Dense cells (C8) compute-bound**: Deploy INT8 (both latency + storage benefit)

**Rejected Methods**:
- ❌ **Pruning**: Not applicable (model already tiny, 77k-500k params)
- ❌ **QAT (Quantization-Aware Training)**: Unnecessary (PTQ accuracy sufficient)
- ❌ **Per-tensor quantization**: Catastrophic accuracy drop (tested, rejected)
- ❌ **QOperator format**: Compatibility issues, reverted to QDQ

---

## 5. EVALUATION METRICS

### 5.1 Classification Metrics
- **Top-1 Accuracy**: Primary metric
- **F1-Score**: Macro-averaged (balanced evaluation)
- **Confusion Matrix**: Per-class error analysis

### 5.2 Biometric Metrics
**Equal Error Rate (EER)**: 
- Scenario: Per-class verification (genuine vs impostor)
- Score: Softmax probability P(class=k | x)
- Compute: EER per class → macro-average
- Threshold: FPR = FNR (equal error point)

**ROC-AUC**:
- Multi-class: One-vs-Rest (OvR), macro-averaged
- Supplementary metric (EER primary untuk biometrik)

### 5.3 Efficiency Metrics

**Model Complexity**:
- Parameters: Count trainable weights
- FLOPs: Multiply-accumulate operations (theoretical)
- Model Size: File size (MB) untuk FP32 dan INT8

**Latency Benchmarking** (Raspberry Pi 5):
- Hardware: Cortex-A76, 4GB RAM, Raspberry Pi OS
- Runtime: ONNX Runtime 1.x CPU (4 threads)
- Optimization: ORT_ENABLE_ALL graph optimization
- Warmup: 20 iterations (discard)
- Measurement: 100 iterations per model
- Statistics: Mean, Median (robust), p95 (tail latency)
- Input: 834 test images (real distribution, bukan synthetic dummy)

**Desktop Benchmarking** (Validation):
- Hardware: MacBook Pro (Apple Silicon)
- Purpose: Sanity check, bukan deployment target

---

## 6. HASIL EKSPERIMEN UTAMA

### 6.1 NAS Search Results

**Lambda Sweep** (hardware-aware penalty):

| Lambda | Dominant Ops         | Akurasi | Latency (Pi) | Trade-off      |
|--------|---------------------|---------|--------------|----------------|
| 0.0    | sep_conv (theory)   | TBD     | TBD          | Accuracy-only  |
| 0.05   | rep_conv+dil+skip   | 97.96%  | 3.99 ms      | **Balanced**   |
| 0.10   | TBD                 | 99.16%  | 6.75 ms      | Moderate speed |
| 0.20   | TBD                 | 99.16%  | 6.29 ms      | Latency-aware  |

### 6.2 Final Model Pareto Frontier (Pi 5, Presisi Optimal)

| Model               | Deploy | Akurasi | Latency | Size   | FLOPs  | Params |
|---------------------|--------|---------|---------|--------|--------|--------|
| **hwNAS λ0.05 C6**  | FP32   | 97.96%  | 3.99 ms | 0.79 MB| TBD    | 315k   |
| hwNAS λ0.20 C8      | INT8   | 98.92%  | 5.27 ms | 0.61 MB| TBD    | 433k   |
| repconv_C8_mid14    | INT8   | 98.92%  | 5.47 ms | 0.60 MB| 130 M  | 503k   |
| mbconv_C6           | FP32   | 99.28%  | 7.16 ms | 0.94 MB| TBD    | 338k   |
| mbconv_C8           | INT8   | 99.28%  | 8.36 ms | 0.87 MB| TBD    | 461k   |
| MobileNetV3-Large   | FP32   | 99.88%  | 15.49 ms| 21 MB  | 233 M  | 5.4M   |

**Key Achievements**:
1. **hwNAS λ0.05 C6 FP32**: 97.96% @ **3.99 ms** (tercepat, acceptable accuracy)
2. **hwNAS λ0.20 C8 INT8**: 98.92% @ **5.27 ms**, 0.61 MB (mendominasi repconv manual)
3. **Size reduction**: 32-35× lebih kecil dari MobileNetV3-Large
4. **Latency speedup**: 2.9-3.9× lebih cepat dari MobileNetV3-Large

### 6.3 NAS vs Manual Design
**Head-to-Head Comparison** (iso-capacity):
- hwNAS λ0.20 C8 INT8 (98.92% @ 5.27 ms) **>** repconv_C8 INT8 (98.92% @ 5.47 ms)
- hwNAS menemukan edge structure lebih optimal (akurasi sama, 0.2 ms lebih cepat)

---

## 7. KONTRIBUSI PENELITIAN

### 7.1 Kontribusi Teknis

**1. Spatial Schedule sebagai Lever Latency Dominan**
- Temuan: stem_downsample=4 vs =2 → 4.4× speedup (akurasi setara)
- Impact: Spatial schedule lebih berpengaruh daripada operator choice untuk latency
- Evidence: mbconv C4 stem=2 (20.46 ms) vs stem=4 (4.69 ms), genotype identik

**2. Operator sebagai Penentu Quantization-Friendliness**
- Mekanisme: Arithmetic intensity (bukan FLOPs/params) menentukan INT8 benefit
- Temuan: Compact cells (skip/dil, low intensity) → INT8 slower; Dense cells (conv padat, high intensity) → INT8 faster
- Evidence: Operator profiling + structural analysis (QDQ node count)

**3. Presisi Deploy = Keputusan Per-Model**
- Guideline: FP32 untuk compact cells (latency), INT8 untuk dense cells (latency+storage)
- Deliverable: Decision rule berbasis architecture properties
- Impact: Challenge "INT8 always faster" assumption pada edge CPU

### 7.2 Kontribusi Metodologi

**1. Hardware-Aware NAS dengan Device LUT**
- Novelty: Latency measurement langsung di Raspberry Pi (bukan proxy FLOPs)
- Benefit: Architecture optimal untuk target hardware spesifik
- Validation: hwNAS λ0.05 C6 mendominasi mbconv C4 manual (accuracy+latency)

**2. P-DARTS untuk Palm Vein NIR**
- Adaptation: 12-op unified search space (sep+dil+mbconv+rep_conv)
- Progressive pruning: 12→7→4 ops (diversity-aware)
- Result: First application of P-DARTS untuk domain biometrik NIR

**3. INT8 PTQ Best Practices untuk Edge**
- Requirement: Per-channel quantization (opset ≥13)
- Pre-processing: Symbolic shape inference + quant_pre_process
- Calibration: Stratified sampling (200 images)
- Impact: Reproducible PTQ pipeline untuk model kecil

### 7.3 Kontribusi Domain (Palm Vein Recognition)

**1. Edge-Deployable Models**
- Achievement: 97-99% accuracy @ <10 ms latency (Pi 5)
- Gap filled: Bridge antara lab accuracy dan real-world deployment
- Baseline: 32× lebih kecil, 3-4× lebih cepat dari MobileNetV3-Large

**2. Comprehensive Baseline Benchmark**
- 9 teacher models evaluated (ResNet, EfficientNet, ConvNeXt, dll.)
- Lightweight baselines (MobileNetV3, ShuffleNetV2, EfficientNetLite)
- Reproducible training protocols (hyperparameters published)

**3. Negative-but-Explained Results**
- KD: Marginal gain pada high-capacity student (headroom terbatas)
- INT8: Architecture-dependent benefit (bukan universal speedup)
- Transparency: Honest reporting sesuai praktik ilmiah yang baik

---

## 8. SOFTWARE DAN TOOLS

**Deep Learning Framework**:
- PyTorch 2.x (training)
- torchvision (teacher models)

**NAS Implementation**:
- Custom P-DARTS (from scratch, adaptasi DARTS asli)
- Genotype representation (DAG structure)

**Deployment**:
- ONNX (model interchange format)
- ONNX Runtime 1.x (inference engine)
- Quantization: onnxruntime.quantization API

**Profiling & Benchmarking**:
- time.perf_counter() (latency measurement)
- Operator-level profiling (isolated ONNX graphs)

**Hardware**:
- Training: GPU/Apple Silicon
- Search: GPU (mixed precision)
- Deployment: Raspberry Pi 5 (CPU-only)

---

## 9. REPRODUCIBILITY

**Seed Control**:
- Global seed: 42
- PyTorch seed, NumPy seed, CUDA seed (deterministic mode)

**Configuration Management**:
- All hyperparameters centralized dalam `nas_config.py`, `kd_config.py`
- Experiment configs saved as JSON (genotype, C_init, num_cells, etc.)

**Data Split**:
- Split info saved as `split_info.json` (train/val/test indices)
- Stratified per-class sampling

**Checkpointing**:
- Best model (validation accuracy)
- Last model (training endpoint)
- Training logs (CSV + TensorBoard-style)

**Documentation**:
- Code comments (inline documentation)
- Markdown reports (FINDINGS.md, EXPERIMENT_SUMMARY.md)
- This document (METODE_EKSPERIMEN_AKTUAL.md)

---

## 10. LIMITATIONS DAN FUTURE WORK

### 10.1 Limitations

**Single Seed**:
- All reported results: single seed (42)
- Head-to-head claims (e.g., hwNAS C6 vs mbconv C4) belum validated dengan multiple seeds + statistical test (McNemar)

**LUT Bias**:
- Isolated operator measurement membayar full QDQ overhead → slight overestimate latency INT8
- Correction applied (floor subtraction) tapi tetap approximation

**NAS bukan Juara Akurasi**:
- hwNAS ~1% di bawah manual mbconv models untuk akurasi maksimal
- Trade-off: NAS unggul di latency, bukan accuracy ceiling

**Quantization Analysis**:
- Mechanism validated qualitatively (operator profiling + node count)
- Quantitative arithmetic intensity calculation (compute/memory ratio) belum implemented

### 10.2 Future Work

**Statistical Validation**:
- ≥3 seeds per model
- McNemar test untuk paired comparison
- Confidence intervals untuk metrics

**Extended Search**:
- λ=0.10/0.20 genotypes (pending deduplication)
- Longer search (50 epochs/stage vs 25)
- Larger search space (mobile operations, attention, etc.)

**Advanced Quantization**:
- Mixed precision (per-layer bit-width)
- QAT (Quantization-Aware Training) untuk extreme compression
- INT4/INT16 exploration

**Larger Dataset**:
- Current: 834 kelas SCUT_PV_v1
- Target: Multi-dataset evaluation (cross-dataset generalization)
- Real-world deployment: live capture, variasi sensor

**Alternative Runtimes**:
- TensorFlow Lite (ARM optimization superior?)
- TVM (auto-tuning compiler)
- Direct ARM NEON kernel (custom implementation)

---

## REFERENSI METODE

**Neural Architecture Search**:
- DARTS: Liu et al., "DARTS: Differentiable Architecture Search", ICLR 2019
- P-DARTS: Chen et al., "Progressive Differentiable Architecture Search", ECCV 2019
- ProxylessNAS: Cai et al., "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware", ICLR 2019

**Knowledge Distillation**:
- Hinton et al., "Distilling the Knowledge in a Neural Network", NeurIPS 2014 Workshop
- FitNets: Romero et al., "FitNets: Hints for Thin Deep Nets", ICLR 2015

**Model Compression**:
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018
- Krishnamoorthi, "Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper", arXiv 2018

**Efficient Networks**:
- MobileNetV2: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
- EfficientNetV2: Tan & Le, "EfficientNetV2: Smaller Models and Faster Training", ICML 2021
- RepVGG: Ding et al., "RepVGG: Making VGG-style ConvNets Great Again", CVPR 2021

**Hardware Efficiency**:
- ShuffleNetV2: Ma et al., "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design", ECCV 2018
- Sze et al., "Efficient Processing of Deep Neural Networks: A Tutorial and Survey", Proceedings of the IEEE 2017

---

**Dokumen ini merepresentasikan metode eksperimen AKTUAL yang digunakan dalam penelitian, bukan proposal awal.**

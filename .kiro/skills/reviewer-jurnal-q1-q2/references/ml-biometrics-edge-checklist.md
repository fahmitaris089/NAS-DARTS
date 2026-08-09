# NAS–KD–Biometrics–Edge Review Checklist

## Contents

1. Biometric data and protocols
2. Architecture search
3. Controlled baseline evaluation
4. Knowledge distillation and compression
5. Quantization and deployment
6. Claims and reporting

## 1. Biometric data and protocols

- State whether the task is verification, closed-set identification, open-set identification, or another protocol.
- Define the identity unit: person, hand, palm, session, image, or sensor capture.
- Audit subject/hand/session leakage rather than filenames only.
- Report classes, subjects, hands, images per class, sessions, devices, exclusions, and source of the available subset.
- Do not speculate about unavailable classes or unpublished dataset restrictions.
- Keep test data out of architecture search, hyperparameter tuning, checkpoint selection, calibration, and baseline selection.
- Report recognition metrics appropriate to the protocol: CRR/accuracy for identification; ROC, EER, FAR/FRR, TAR at fixed FAR, or DET for verification as appropriate.
- Do not compare verification EER directly with closed-set classification accuracy.
- If one image per class is tested, report absolute errors and treat sub-percentage differences cautiously.
- Limit generalization claims when only one dataset, sensor, population, or split is evaluated.

## 2. Architecture search

- Define the search space, cell topology, candidate operators, stage schedule, optimization, regularization, latency term, and discretization rule.
- Report search data partitions and ensure search validation is not the held-out test set.
- Report search cost, device profiling cost, software/runtime, and whether search was repeated.
- Validate the latency lookup table against end-to-end model latency and discuss estimation error.
- Treat the LUT as device/runtime/precision-specific.
- Compare against a simpler manually designed architecture and established NAS architectures.
- Test whether the selected genotype improves the stated trade-off, not merely accuracy under a different training recipe.
- Separate architecture-search contribution from retraining tricks.

## 3. Controlled baseline evaluation

- Use the same random initialization policy, split, preprocessing, augmentation, optimizer, scheduler, budget, checkpoint rule, and test policy for architecture-only comparisons.
- Report scratch and ImageNet-pretrained results in separate tables.
- Do not give the proposed architecture more epochs, KD, augmentation, or test-set tuning than baselines while claiming architectural superiority.
- Use multiple seeds for stochastic training; report each seed and mean ± sample SD.
- Explain seed choice as predetermined reproducibility control, not evidence of robustness by itself.
- Include domain-specific palm-vein models and general lightweight/hardware-aware baselines for different scientific roles.
- Label paper-constrained reconstructions and document deviations from the publication.
- Do not copy reported literature scores into the controlled table.

## 4. Knowledge distillation and compression

- Fix teacher checkpoint, teacher evaluation, temperature, hard/soft loss weights, student budget, and selection rule.
- Compare each distilled student with its own non-distilled counterpart.
- Apply KD to selected strong baselines if claiming that the proposed architecture benefits uniquely from KD.
- Keep teacher selection independent of the held-out test set.
- Report whether teacher and student use logits, features, attention maps, or another signal.
- Separate pruning-only, KD-only, joint compression, and fine-tuning effects.
- Report sparsity structure and whether theoretical sparsity produces runtime improvement.

## 5. Quantization and deployment

- State PTQ versus QAT, graph format, quantization representation, weight/activation types, per-channel/per-tensor policy, calibration method, and excluded operators.
- Use training-only calibration data with a fixed manifest across models.
- Evaluate FP32 and INT8 accuracy on the same test manifest.
- Distinguish PyTorch, exported ONNX, ONNX FP32, and ONNX INT8 results.
- Validate export numerically before quantization.
- Report device model, CPU architecture, OS, runtime/version, power mode, thread counts, execution mode, batch size, warm-up, timed iterations, and thermal control.
- Report mean, median, and tail latency such as p95; include variability when runs are repeated.
- Measure model file size and, when relevant, peak memory or energy; do not infer latency solely from FLOPs or parameter count.
- Benchmark all models under identical runtime settings and input preparation boundaries.

## 6. Claims and reporting

- Define the metric behind `best`, `lightweight`, `real-time`, `robust`, and `hardware-aware`.
- Use Pareto dominance only when all stated dimensions and measurement conditions support it.
- If another model is more accurate but slower/larger, describe the constraint-dependent trade-off.
- If one comparator dominates accuracy, size, latency, and complexity, address the lack of practical advantage directly.
- Separate controlled findings from contextual literature and deployment results.
- State limitations for subset availability, split/session metadata, reconstruction fidelity, dataset count, device count, and search repetition.
- Preserve exact provenance for every checkpoint, configuration, split manifest, and reported metric.

# C10 PK–Adaptive Center–Relation Screening Protocol

Implementation label:

> Adaptive Center–Relation Distillation (inspired by AdaDistill and CoupleFace)

This is an independent adaptation. It is not an official implementation or an
exact reproduction of AdaDistill or CoupleFace. The method retains the C10
closed-set CE classifier for inference. The projection adapter and teacher
center bank are training-only and are excluded from ONNX export.

Run the validation-only audit and four seed-42 ablations from the repository
root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_c10_representation_kd.ps1 -Mode audit
powershell -ExecutionPolicy Bypass -File scripts/run_c10_representation_kd.ps1 -Mode pk_ce
powershell -ExecutionPolicy Bypass -File scripts/run_c10_representation_kd.ps1 -Mode center
powershell -ExecutionPolicy Bypass -File scripts/run_c10_representation_kd.ps1 -Mode hybrid_scratch
powershell -ExecutionPolicy Bypass -File scripts/run_c10_representation_kd.ps1 -Mode hybrid_early
```

Add `-Smoke` to `pk_ce`, `center`, `hybrid_scratch`, or `hybrid_early` for a
one-epoch integration check before launching the full run. The early smoke test
still requires the full E1 epoch-100 checkpoint because its initialization is
part of the experimental definition.

All modes are screening runs and use validation-only checkpoint selection.
Do not evaluate the test split until one configuration has been frozen. E1–E3
reuse the same hashed random initial state. E4 starts from the E1 epoch-100
student weights and is reported as a weights-only continuation.

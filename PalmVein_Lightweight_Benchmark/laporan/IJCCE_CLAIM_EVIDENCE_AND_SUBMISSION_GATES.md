# IJCCE Claim-Evidence Ledger and Submission Gates

This companion file audits the working manuscript `IJCCE_Manuscript_Sections_1_3_Source.md`. It is not part of the submitted article.

## Evidence boundaries already enforced

| Claim area | Defensible wording | Evidence location | Status |
|---|---|---|---|
| Task | Closed-set 834-class palm-vein identification | `configs/dataset.json`, `src/evaluation/metrics.py` | Locked |
| Outer split | 6,672/834/834, zero exact filename overlap | `dataset/splits/split_info.json`, `src/data/dataset.py` | Locked; image-level only |
| Dataset provenance | 834 identities/8,340 images supplied by owner | Thesis and dataset setup notes | State without guessing why 266 identities are absent |
| Preprocessing | Offline 384 ROI, CLAHE, min-max normalization, 224 resize | `Eksperimen_Hardware_Aware_PDARTS/src/preprocessing/preprocessing.py` | Use archived implementation, not benchmark reference helper |
| NAS search | 12 operators; stages 5/8/11; 25 epochs each; 12/7/4 ops | `src/nas/nas_config.py`, `configs/search/thesis_lambdas.json` | Locked |
| LUT construction | Up to 60 operator-shape probes; ONNX/QDQ conversion; device timing; corrected and aggregated operator costs | `Eksperimen_Hardware_Aware_PDARTS/src/latency_lut/`, archived LUT artifacts | Figure 3; resource prior only, not full-model latency |
| NAS latency objective | Max-normalized expected edge cost, averaged over normal/reduction cells | `src/nas/architect.py` | Figure 4; locked |
| Controlled scratch | 600 epochs x seeds 42/123/2026, AdamW, no KD/pretraining | `configs/scratch_600e.json` | Locked |
| Pretrained transfer | 200 epochs x three seeds for three official-weight models | `configs/pretrained_200e.json`, model provenance | Locked for current registry |
| Test metrics | Accuracy/correct/errors; sample SD across seeds | `src/evaluation/metrics.py`, `scripts/summarize_results.py` | Locked |
| Full-model PTQ | QDQ, QInt8 per-channel weights, QUInt8 activations, MinMax | `configs/deployment.json`, `scripts/quantize_int8.py` | Locked |
| Target timing | Batch 1, 4/1 threads, 50 warm-up, 500 timed, mean/median/p95/min/max | `configs/deployment.json`, `scripts/benchmark_raspberry_pi.py` | Locked |
| Equation provenance | Ten numbered equations encode the implemented DARTS, LUT, KD, accuracy, and seed-summary definitions | `IJCCE_Manuscript_Sections_1_3_Source.md`, `src/nas/architect.py`, training/evaluation code | Native editable OMML in v6; verify against code again after protocol freeze |
| Literature support | 35 claim-linked sources, numbered by first appearance for the working draft | DOI/publisher/ISO/PMLR/OpenReview/CVF/official documentation records | Citation and reference sets match; convert to APA author-year before submission |

## Fatal submission gates

1. **Legacy test exposure.** The same test split was used throughout thesis exploration. `configs/teacher/final_teacher.json` explicitly lists `test_accuracy` among teacher-selection criteria, and thesis candidate selection compared lambda/capacity/KD settings after test evaluation. Results using this split cannot be called untouched confirmatory evaluation. Options:
   - create a newly sealed, preferably session-disjoint test partition and rerun all compared models; or
   - keep the current split and label the evidence retrospective/descriptive, with an external dataset used for stronger confirmation.

2. **Teacher and KD selection.** The journal-level KD table cannot reuse a teacher or `(T, alpha)` chosen using legacy test outcomes as if it were validation-only. Freeze the teacher from validation loss and select KD hyperparameters without test access, then run P-DARTS and at least two baselines under identical settings.

3. **LUT/deployment quantizer mismatch.** `quantize_lut_probes.py` uses signed `QInt8` activations, whereas full-model deployment uses `QUInt8` activations. Regenerate the LUT with the deployed quantizer or supply an explicit sensitivity experiment. Until then, claim “INT8-informed hardware-aware search,” not “identical deployment-recipe LUT.”

4. **Raspberry Pi environment ledger.** Add Raspberry Pi model and RAM, OS/kernel, CPU governor, cooling, power state, ONNX Runtime version, thread affinity, and raw per-inference timing file. Host timing must never be labeled Raspberry Pi timing.

## Major revision gates

1. **Session and near-duplicate audit.** The manifest contains no session identifiers. Ask the data owner whether acquisition-session metadata exist. If unavailable, retain the image-level split and limitation; do not claim cross-session generalization.

2. **Domain comparator status.** `src/models/chen.py` and its JSON config remain as audit artifacts but are intentionally unregistered. The controlled domain comparison uses PalmNet as a paper-constrained independent reconstruction. Do not report Chen training results in the benchmark table.

3. **AMPVNet status.** Do not list AMPVNet in the controlled result table until an auditable implementation has been adapted and trained by the common engine. Literature values remain contextual.

4. **Ding reconstruction wording.** Always use “paper-constrained independent reconstruction.” Agreement with the reported six-block structure and final channel table does not establish performance equivalence to unavailable author code.

5. **Remaining FP32 nodes.** Enumerate operator types that remain unquantized in every final ONNX graph. QDQ conversion alone is not evidence of fully integer execution.

6. **FLOPs/MMAC convention.** Freeze the counting tool, input `[1,3,224,224]`, and whether one MAC is reported as one or two FLOPs.

## IJCCE format gates before submission

- Convert all numeric `[n]` citations to APA author-year and sort references alphabetically.
- Add authors, affiliations, corresponding-author details, factual abstract, and keywords.
- Add CRediT roles, competing-interest declaration file, funding statement, data availability, acknowledgements, and any required AI-use declaration.
- Keep the manuscript single-column and simply formatted.
- Keep tables editable with no vertical rules or cell shading.
- Upload every final figure separately at the required artwork resolution and keep captions in the manuscript.
- Prepare 3-5 optional highlights in a separate editable file, each no more than 85 characters including spaces.

## Figure production ledger

| Figure | Thesis source | Required change |
|---|---|---|
| Fig. 1 | Fig. 3.1 | Add exact partition roles, nested NAS split, three protocol branches, training-only calibration, and legacy-test boundary |
| Fig. 2 | Figs. 3.2-3.3 | Align labels with archived preprocessing code: 384 ROI and Lanczos resize |
| Fig. 3 | Fig. 3.5 | Limit the diagram to operator-shape probes, QDQ conversion, Raspberry Pi timing, raw/corrected costs, and aggregation into the LUT |
| Fig. 4 | Fig. 3.8 | Show LUT normalization, classification and resource paths, the combined objective, and the gradient path to architecture parameters |
| Fig. 5 | Fig. 3.10 | Add parity gate, manifest hash, FP32/INT8 accuracy branches, and timing statistics |

## Reference conversion gate

The v6 manuscript uses 35 temporary numbered citations by author request. The numbering follows first appearance and the in-text citation set matches the reference set exactly. DOI-bearing records have unique DOI strings; sources without a DOI use an authoritative publisher, standards-body, proceedings, or official documentation URL with an access date where appropriate. Before submission, import and independently verify all records in Mendeley, switch to IJCCE/APA author-year style, and confirm the one-to-one citation/reference match again. Do not submit the numbered draft.

## Equation conversion gate

The v6 builder accepts only `[[EQ:eq_id]]` and `[[MATH:math_id]]` source markers from locked registries. It raises `ValueError` for unknown display or inline identifiers and for a missing, duplicated, or reordered display equation. The generated DOCX must retain exactly ten `m:oMathPara` objects for Eqs. (1)-(10), editable inline `m:oMath` objects, and no unconverted markers or image-based equations. Any later mathematical change must be made in the source and OMML registry rather than typed over the rendered DOCX.

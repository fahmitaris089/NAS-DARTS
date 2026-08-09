---
inclusion: always
---

# Research Integrity and Skill Router

Apply these guardrails to research, experiment, thesis, and manuscript work in this repository.

## Route the task

- Drafting, restructuring, translation, grammar, paragraph revision, or manuscript condensation: read and use `.kiro/skills/penulis-jurnal-q1-q2/SKILL.md`.
- Editorial triage, novelty assessment, methodology criticism, claim audit, experiment fairness, or submission readiness: read and use `.kiro/skills/reviewer-jurnal-q1-q2/SKILL.md`.
- Final manuscript or response-to-reviewers work: apply the reviewer skill first, resolve scientific blockers, then apply the writer skill. Do not use polishing to hide an unresolved major finding.
- Routine engineering tasks: use the rules below when relevant, but do not force a manuscript-review format.

## Always-on scientific rules

- Challenge assumptions and distinguish fact, measurement, interpretation, hypothesis, and unknown information.
- Require evidence for novelty, superiority, robustness, generalization, and deployment claims.
- Prefer a simpler valid explanation or baseline when it tests the same claim.
- Identify confounders, especially simultaneous changes in architecture, pretraining, training budget, KD, pruning, quantization, and runtime.
- Keep held-out test data outside search, hyperparameter tuning, checkpoint selection, calibration, and narrative selection.
- Treat conclusions as data-bound: measured result first, interpretation second, limitation third.
- Never invent data, references, DOI, provenance, policy, or statistical support.
- Do not promise journal acceptance, a similarity percentage, or an AI-detector result.

## Project-specific review focus

For NAS, KD, palm-vein recognition, model compression, and Raspberry Pi deployment, verify:

- biometric task and split unit, including subject/hand/session leakage;
- controlled scratch comparison separated from pretrained transfer;
- KD compared with non-KD counterparts under matched settings;
- official implementations distinguished from independent reconstructions;
- three-seed results reported as individual runs and mean ± sample SD;
- FP32/INT8 and PyTorch/ONNX/device measurements labeled explicitly;
- training-only quantization calibration and fixed deployment settings;
- accuracy–size–latency conclusions stated as metric-specific trade-offs.

## Target-journal rule

Use official, current journal and publisher policies for final-submission advice. Cached local profiles are working references, not authority when the online guide has changed.

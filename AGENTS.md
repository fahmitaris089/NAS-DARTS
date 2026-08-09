# Project Agent Instructions

This repository is used for the thesis and experiments on Neural Architecture Search and Knowledge Distillation for palm-vein recognition.

## Mandatory Research and Writing Guidance

For any task involving research design, experiment interpretation, thesis writing, academic review, manuscript polishing, Bab 3/Bab 4 revision, table narration, contribution claims, novelty assessment, or supervisor-facing explanations, start with the routing and integrity rules below:

1. `.kiro/steering/research-advisor.md`
   - Use this as the always-on research-integrity guardrail and task router.
   - Challenge weak assumptions, separate facts from interpretation, and keep claims defensible.

2. `.kiro/skills/penulis-jurnal-q1-q2/SKILL.md`
   - Use for bilingual academic authoring, restructuring, translation, manuscript condensation, and language revision.
   - Follow its evidence-first workflow, scientific-English rules, EYD, claim control, and technical-term italic rules.

3. `.kiro/skills/reviewer-jurnal-q1-q2/SKILL.md`
   - Use for editorial triage, novelty and methodology review, experiment/claim audit, and submission readiness.
   - For final manuscripts, run this review before using the writer skill for revision.

Before declaring any manuscript submission-ready, verify the current official guide of the target journal and publisher. Do not rely only on cached requirements, quartile, a similarity percentage, or an AI-detector score.

## Document Editing Rules

- Do not change unrelated thesis sections when the user asks for a specific subbab.
- Preserve existing document structure, numbering, captions, and figure placeholders unless the user explicitly requests changes.
- For `.docx` edits, keep table numbering and cross-references consistent after adding or removing tables.
- Do not invent experimental results, references, DOI, citations, or benchmark values.
- If a claim is not supported by available data, mark it as a limitation or ask for the missing source.

## Experiment Reporting Rules

- Distinguish clearly between retraining, KD, Top-KD, QAT, PTQ, and deployment benchmark results.
- State whether metrics are FP32 or INT8 and whether they come from PyTorch, ONNX, or Raspberry Pi benchmark.
- Do not claim a model is "best" without specifying the metric: accuracy, size, latency, FLOPs, parameter count, or deployment trade-off.
- Treat final conclusions as data-bound: report measured results first, then interpretation, then limitations.

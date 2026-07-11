# Project Agent Instructions

This repository is used for the thesis and experiments on Neural Architecture Search and Knowledge Distillation for palm-vein recognition.

## Mandatory Research and Writing Guidance

For any task involving research design, experiment interpretation, thesis writing, academic review, manuscript polishing, Bab 3/Bab 4 revision, table narration, contribution claims, novelty assessment, or supervisor-facing explanations, always apply both local instruction sources below:

1. `.kiro/steering/research-advisor.md`
   - Use this as the research-quality and critical-review persona.
   - Challenge weak assumptions, separate facts from interpretation, and keep claims defensible.
   - Use it especially when evaluating NAS, KD, quantization, Raspberry Pi deployment, SOTA comparison, ablation logic, or publication positioning.

2. `.kiro/skills/penulis-jurnal-q1-q2/SKILL.md`
   - Use this as the Indonesian academic writing and thesis-editing standard.
   - Follow its EYD, academic tone, table/caption narration, claim-control, and technical-term italic rules.
   - Before editing thesis text, check the section on `Aturan Italic Istilah Teknis`.

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

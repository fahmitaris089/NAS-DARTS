# Scientific Writing and Original Expression

## Contents

1. Evidence-first prose
2. Paragraph design
3. Scientific English
4. Academic Indonesian
5. Section-level tense and voice
6. Formulaic-language audit
7. Final language checklist

## 1. Evidence-first prose

Natural academic writing begins with a specific intellectual task, not stylistic variation. Before drafting a paragraph, identify:

- the question answered;
- the evidence available;
- the inference permitted by that evidence;
- the boundary beyond which the claim would be speculative.

Write from an evidence note rather than copying the syntax of a source. When a source is required, record the claim it supports and verify the full source before drafting. Avoid paraphrasing an abstract as though the full method and limitations had been checked.

Use concrete anchors whenever available: dataset, population, model, comparison, metric, uncertainty, device, runtime, or table/figure. Remove sentences that merely announce importance without adding evidence.

## 2. Paragraph design

Give each paragraph one dominant rhetorical role. A paragraph may:

- define the local problem;
- synthesize a group of studies;
- expose a methodological limitation;
- justify a design choice;
- report an observation;
- interpret a result;
- qualify a claim;
- connect a finding to deployment.

The first sentence should orient the reader to that role. The remaining sentences should develop it rather than repeat it with synonyms. End when the function is complete; do not force a generic concluding sentence.

Vary paragraph and sentence length as a consequence of reasoning. Do not manufacture variation to influence an AI detector. Short sentences can state a result or limitation. Longer sentences can express a conditional comparison, provided their grammar remains controlled.

Use transitions that name the relationship:

- contrast: `However`, `In contrast`, `Unlike the controlled setting`;
- condition: `Under the same training budget`, `For the INT8 graph`;
- consequence: `This difference limits`, `Consequently`;
- qualification: `Within this dataset`, `This interpretation remains tentative because`.

Do not begin several consecutive paragraphs with the same template.

## 3. Scientific English

### Dialect

Use American English for a new IJCCE manuscript. If an existing manuscript consistently uses British English, preserve it. Never mix pairs such as `behavior/behaviour`, `analyze/analyse`, or `optimization/optimisation` without reason.

### Grammar pass

Audit each sentence for:

- subject–verb agreement;
- article use (`a`, `an`, `the`, or zero article);
- singular/plural consistency;
- pronoun antecedents;
- parallel items in lists and comparisons;
- misplaced or dangling modifiers;
- punctuation around dependent clauses;
- consistent mathematical symbols and units;
- unnecessary nominalization;
- ambiguous references such as `this`, `it`, or `they`.

Do not automatically replace active voice with passive voice. Use active voice when the actor or decision matters (`We fixed the split before training`). Use passive voice when the procedure or object is the focus (`The checkpoint was selected by validation loss`).

Prefer restrained verbs:

- observation: `reached`, `decreased`, `required`, `differed`;
- interpretation: `suggests`, `is consistent with`, `may reflect`;
- unsupported causality: do not use `caused`, `led to`, or `demonstrates` unless the design supports it.

### Precision

Replace vague comparisons with scoped statements:

- weak: `The proposed model performed better.`
- precise: `The proposed model reduced median INT8 latency on the Raspberry Pi 5 while retaining comparable three-seed accuracy.`

State the denominator, unit, precision, and evaluation surface when ambiguity is possible.

## 4. Academic Indonesian

Use Indonesian that follows EYD, with direct sentence structure and stable terminology. Avoid literal translations that obscure technical meaning.

Use Indonesian equivalents when natural, for example `akurasi`, `ukuran model`, `waktu inferensi`, and `kuantisasi`. Retain standard English terms when their translation would be awkward or ambiguous.

Italicize foreign technical phrases in Indonesian prose, including *hardware-aware*, *latency*, *lookup table*, *trade-off*, *checkpoint*, *inference*, *knowledge distillation*, *teacher model*, *student model*, and *post-training quantization*.

Do not italicize:

- acronyms and formats: CNN, NAS, KD, PTQ, QAT, INT8, FP32, ONNX, FLOPs;
- model, hardware, and software names: P-DARTS, MobileNetV3, Raspberry Pi, PyTorch, ONNX Runtime;
- configuration identifiers and paths: `batch_size`, `weight_decay`, `lr_min`, filenames, and directories.

Keep the same technical term and italic treatment throughout one paragraph, caption, or edited section. Table cells may prioritize compact readability, but captions and narrative follow the prose rules.

## 5. Section-level tense and voice

- Established knowledge: present tense.
- Completed procedures: past tense.
- Observed experimental results: past tense.
- Meaning that remains valid at reading time: present tense, used cautiously.
- Planned or unfinished experiments: do not present as completed; mark explicitly.

Avoid first person only when the journal or author preference requires it. First-person plural can clarify author decisions, but it must not be used to inflate claims.

## 6. Formulaic-language audit

Revise repeated or empty constructions such as:

- `In today's rapidly evolving landscape`;
- `It is worth noting that`;
- `It is important to highlight that`;
- `The results clearly demonstrate`;
- `This groundbreaking approach`;
- `Berdasarkan hasil tersebut` in every paragraph;
- `Dapat dilihat bahwa` when the sentence can state the observation directly;
- `Hal ini menunjukkan bahwa` without naming what `hal ini` refers to.

These expressions are not forbidden individually. Revise them when they replace evidence, repeat a paragraph pattern, or overstate certainty.

Do not use deliberate errors, random colloquialisms, awkward synonyms, or sentence scrambling as “humanization.” Do not report or target an AI-detector score.

## 7. Final language checklist

- Every paragraph has a discernible scientific function.
- Each major claim points to data or a verified source.
- Observation, interpretation, and implication remain distinct.
- Grammar edits have not strengthened the claim.
- Terminology, abbreviations, symbols, dialect, and units are consistent.
- No sentence is retained only because it sounds polished.
- No placeholder is presented as final evidence.

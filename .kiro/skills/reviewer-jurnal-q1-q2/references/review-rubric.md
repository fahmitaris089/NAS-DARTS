# Q1/Q2 Manuscript Review Rubric

## Contents

1. Evidence rules
2. Editorial triage
3. Peer-review dimensions
4. Severity and readiness
5. Review report quality

## 1. Evidence rules

Base every finding on a manuscript location, supplied artifact, verified source, or explicit absence. Mark unavailable evidence as `unknown`; do not silently convert it to pass or fail.

Distinguish:

- measured fact;
- author interpretation;
- reviewer inference;
- policy requirement;
- optional recommendation.

Do not treat reviewer preference as a methodological defect. Explain how a recommendation changes validity, reproducibility, relevance, or clarity.

## 2. Editorial triage

### Scope and article type

- Does the problem fit the stated aims of the journal?
- Is the engineering or scientific relevance explicit?
- Is the manuscript the claimed article type?
- Does the title/abstract accurately represent the actual evaluation?

### Originality and significance

- Is the closest prior work identified and compared at the level of method and evidence?
- Is the novelty more than combining familiar components?
- Does the contribution change knowledge, methodology, evidence, or deployment practice?
- Could the same conclusion follow from a simpler baseline not tested?

### Ethics and integrity

- Are data rights, consent/privacy, authorship, conflicts, funding, permissions, and AI use addressed?
- Are results, references, and provenance auditable?
- Is thesis/preprint/conference overlap transparent?
- Are any test-set, image-manipulation, or reporting practices disqualifying?

### Minimum methodological validity

- Can the design answer the stated research question?
- Is the unit of analysis correct?
- Are train, validation, test, and calibration roles separated?
- Are confounders controlled sufficiently for the claimed causal interpretation?

### Minimum communication quality

- Can an editor identify problem, gap, contribution, method, main result, and limitation from the manuscript?
- Are placeholders, contradictory numbers, missing captions, or broken references present?
- Is the English comprehensible and internally consistent?

## 3. Peer-review dimensions

### Research question and contribution

- The objective is testable and stable across title, abstract, introduction, methods, results, and conclusion.
- Contributions are specific and mapped to evidence.
- Novelty claims cite and distinguish the closest work.

### Data and protocol

- Dataset source, inclusion, exclusions, licenses, classes/subjects, sessions, and sample counts are reported.
- Split construction prevents leakage at the correct identity/session level.
- Preprocessing and augmentation are fixed before test evaluation.
- External validity matches the number of datasets, devices, and populations evaluated.

### Methods and reproducibility

- Architecture, algorithms, losses, hyperparameters, seeds, software, hardware, and selection rules are reproducible.
- Official implementation, adaptation, and independent reconstruction are clearly distinguished.
- The method does not depend on undisclosed trial-and-error with the test set.

### Comparators and ablations

- Baselines answer distinct scientific questions and are not deliberately weakened.
- Training budget, initialization, preprocessing, augmentation, and selection rules are controlled where architecture claims are made.
- Pretraining, KD, pruning, and quantization effects are reported separately unless the claim concerns the complete system.
- Ablations test the claimed contribution rather than arbitrary component removal.

### Statistics and uncertainty

- Repeated training reports completed run count, central tendency, and variability.
- Sample standard deviation uses the appropriate denominator for seed summaries.
- Statistical tests, confidence intervals, or effect sizes are used only when their assumptions and unit of analysis are defensible.
- Small differences are not overstated, especially with one test sample per class or few seeds.

### Results and interpretation

- Tables and figures match the text and labels identify precision/runtime/device.
- Results answer the research questions without selective reporting.
- Discussion explains significance, plausible mechanisms, competing explanations, and limitations.
- Causal language matches the design.
- Literature values from incompatible protocols are not placed in a controlled ranking.

### Practical relevance

- Efficiency is measured on the intended runtime/device rather than inferred solely from parameters or FLOPs.
- Accuracy, size, latency, memory, energy, and calibration effects are reported according to the claim.
- Pareto or constraint-based arguments are preferred over unqualified “best” claims.

### Writing and presentation

- Each section performs its scientific function.
- Terminology, notation, units, citations, captions, and cross-references are consistent.
- The manuscript is concise without omitting reproducibility details.
- Abstract and conclusion do not contain unsupported or provisional results.

## 4. Severity and readiness

Use `fatal` only when the study or submission cannot ethically or logically support its central claim. Use `major` for deficiencies likely to affect an editorial decision or conclusion. Use `minor` for local correction.

Readiness rules:

- `not ready`: any confirmed fatal finding, unresolved integrity issue, or missing central evidence;
- `major revision`: at least one major finding that can plausibly be repaired;
- `near-ready`: no fatal finding and no unresolved validity-threatening issue; remaining work is bounded;
- `submission candidate`: all available evidence reviewed, target checklist complete, and no major finding remains.

Never convert these bands to acceptance probabilities.

## 5. Review report quality

A useful report:

- states what was actually reviewed;
- identifies the strongest supported contribution;
- gives precise, evidence-linked criticism;
- separates required changes from optional improvements;
- proposes claim restriction when more experiments are disproportionate;
- avoids hostility, prestige signaling, and vague demands;
- never invents citations or claims knowledge of unavailable files.

# FinQA Dataset

This document summarizes how this project interprets the FinQA dataset.

FinQA is a financial question-answering dataset built around mixed evidence from report text and tables. Many examples require selecting the right evidence and performing arithmetic, so the project treats FinQA as a staged reasoning task rather than ordinary extractive QA.

## Why FinQA Fits This Project

FinQA examples can require the system to:

- find the correct sentence, row, year, or metric
- combine evidence from text and tables
- perform arithmetic or comparisons
- handle financial formatting, units, percentages, and rounding
- explain where failures happen across retrieval, prompting, parsing, execution, and evaluation

That makes it a good fit for this pipeline:

1. load and normalize examples
2. construct evidence units
3. retrieve relevant evidence
4. prompt a model with constrained context
5. parse a direct answer or executable program
6. execute arithmetic deterministically
7. evaluate final answers and retrieval support

## Dataset Splits

The expected raw files are:

- `train.json`
- `dev.json`
- `test.json`
- `private_test.json`

Practical use in this project:

- `train.json`: useful for understanding examples and running experiments
- `dev.json`: useful for iteration and model/prompt checks
- `test.json`: useful for public-style evaluation
- `private_test.json`: useful as an inference-only shape with missing gold labels

The loader supports labeled records and private-test-style records where evaluation labels are absent.

## Record Structure

Each FinQA example is a JSON object with document content and a nested `qa` object.

Common top-level fields:

- `id`
- `filename`
- `pre_text`
- `post_text`
- `table`
- `table_ori`
- `qa`

The `qa` object usually contains:

- `question`
- `answer`
- `exe_ans`
- `program`
- `program_re`
- `gold_inds`

Some splits also include extra retrieval or debug fields such as `model_input`, `tfidftopn`, `text_retrieved`, and `table_retrieved`.

## Runtime Inputs

Runtime inputs are fields the system may use during inference:

- `id`
- `filename`
- `pre_text`
- `post_text`
- `table`
- `qa.question`

These become `RuntimeInputs` in `src/data/schemas.py`.

## Gold Targets

Gold targets are labels used only for evaluation or analysis:

- `qa.answer`
- `qa.exe_ans`
- `qa.program`
- `qa.program_re`
- `qa.gold_inds`

These become `GoldTargets` in `src/data/schemas.py`.

The runtime pipeline should not use these fields to answer a question.

## Metadata

The project preserves optional dataset artifacts as metadata when useful:

- original table variants
- dataset-provided retrieval/debug fields
- annotation helpers
- explanation or step fields when present

Metadata is useful for inspection, but it should not be treated as model input unless a stage explicitly opts into it.

## Important Fields

### `pre_text` And `post_text`

These are lists of sentence-like strings before and after the table. Evidence construction preserves the original items as text evidence units so ids can align with FinQA support labels.

### `table`

The table is represented as a nested list of strings. Evidence construction renders rows into text-like evidence units such as:

```text
the 2015 net revenue of amount ( in millions ) is $ 5829 ;
```

### `qa.question`

The natural-language question. This is the main input to retrieval and prompting.

### `qa.exe_ans`

The gold executable answer. This is the primary final-answer evaluation target when available.

### `qa.program`

The gold reasoning program. This is useful for parser tests and optional reasoning analysis, but it is not used during inference.

### `qa.gold_inds`

Gold supporting evidence ids. These are useful for evaluating retrieval directly.

## Internal Representation

The normalized internal example is split into:

- `RuntimeInputs`
- `GoldTargets`
- `ExampleMetadata`

This separation is deliberate. It prevents label leakage and makes the inference/evaluation boundary easy to explain in the report.

## Related Docs

- [01. Data Loading](../stages/01_data_loading.md)
- [02. Evidence Construction](../stages/02_evidence_construction.md)
- [03. Retrieval](../stages/03_retrieval.md)

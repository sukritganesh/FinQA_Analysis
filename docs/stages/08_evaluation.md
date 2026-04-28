# 08. Evaluation

## Purpose

Compare predictions against FinQA gold targets when labels are available. Evaluation is optional so the same pipeline can run on private-test-style data with no gold answers.

## What This Stage Does

- Compares `Prediction.answer` to `example.gold.executable_answer` when available.
- Handles missing predictions and missing gold answers.
- Uses exact matching for yes/no answers.
- Uses tolerant numeric matching for numeric answers.
- Writes JSON and Markdown summaries for batch runs.
- Keeps retrieval evaluation separate but compatible.

## Key Files

- `src/eval/metrics.py`
- `src/eval/answer.py`
- `src/eval/retrieval.py`
- `scripts/evaluate_retrieval.py`
- `scripts/run_pipeline.py`
- `tests/test_answer_evaluation.py`
- `tests/test_retrieval_eval.py`

## Answer Matching

Gold executable answers in the labeled FinQA splits are clean strings. Most are numeric; yes/no answers are represented as `yes` or `no`.

Evaluation rules:

- If gold is missing, skip correctness scoring.
- If prediction is missing and gold exists, mark the example incorrect.
- Compare yes/no answers after lowercase normalization.
- Parse numeric answers and compare with absolute or relative tolerance.
- Allow percent-like direct answers only as a forgiving evaluation normalization; the model-output parser still rejects percent signs in programs.

Numeric tolerance:

```text
absolute_tolerance = 1e-4
relative_tolerance = 1e-4
```

## Batch Outputs

Pipeline runs write compact predictions:

```text
reports/test_results/<run_name>/predictions.json
```

When evaluation is enabled, the runner also writes:

```text
reports/test_results/<run_name>/<run_name>_summary.json
reports/test_results/<run_name>/<run_name>_details.json
reports/test_results/<run_name>/<run_name>_details.md
```

The summary contains counts such as total examples, examples with gold, answered examples, correct answers, accuracy, missing predictions, and pipeline error counts.

The details files preserve one row per example with prediction, gold answer, match result, and errors.

## Retrieval Evaluation

Retrieval has its own evaluation path against FinQA support ids. This is useful because answer failures can come from different places:

- retrieval missed the needed evidence
- the model produced an unparsable output
- deterministic execution failed
- the final answer was numerically wrong

Keeping retrieval evaluation separate makes those failure modes easier to isolate.

## Boundaries

Evaluation should not:

- call the model
- rerun retrieval
- change predictions
- use gold labels during inference

Evaluation should:

- score existing predictions
- preserve enough detail for error analysis
- make results reproducible from files

## Verification

Tests cover:

- exact yes/no matching
- numeric tolerance
- percent normalization for direct answers
- missing predictions
- missing gold answers
- batch summaries
- output file writing
- retrieval support comparison

## Possible Improvements

- Add optional program similarity metrics.
- Join retrieval hits with answer details for stage-level error analysis.
- Add summarized error categories for report tables.

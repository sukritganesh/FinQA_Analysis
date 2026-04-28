# 01. Data Loading

## Purpose

Load FinQA JSON records into a stable internal representation. This stage defines the boundary between data that is safe to use at inference time and labels that are only valid for evaluation.

Dataset fields and split behavior are summarized in [FinQA Dataset](../details/finqa_dataset.md).

## What This Stage Does

- Reads FinQA JSON files from disk.
- Normalizes raw records into `FinQAExample` objects.
- Stores runtime inputs separately from gold labels.
- Preserves useful dataset metadata for debugging.
- Supports both labeled splits and private-test-style records with missing gold answers.

## Key Files

- `src/data/schemas.py`
- `src/data/loader.py`
- `src/graph/data_loading.py`
- `scripts/inspect_example.py`
- `tests/test_loader_and_schemas.py`
- `tests/test_data_loading_graph.py`

## Normalized Shape

Each example is split into three sections:

- `RuntimeInputs`: example id, filename, question, `pre_text`, `post_text`, and table.
- `GoldTargets`: answer, executable answer, gold program, nested program, support facts, and annotation helpers.
- `ExampleMetadata`: original table and dataset-provided retrieval/debug artifacts.

This keeps inference code from accidentally using gold labels.

## Important Behavior

- `train.json`, `dev.json`, and `test.json` include gold labels.
- `private_test.json` style records are valid even when labels are missing.
- `qa.answer` and `qa.exe_ans` are both preserved because they serve slightly different purposes.
- Extra FinQA fields such as `model_input`, `tfidftopn`, `text_retrieved`, and `table_retrieved` are kept as metadata, not inference inputs.

## Graph Integration

The data-loading graph can load a split and select one example by index or id:

```text
START -> load_examples -> select_example -> END
```

The full pipeline normally receives a selected `FinQAExample` and continues from there.

## Verification

Tests cover:

- runtime/gold separation
- labeled examples
- private-test-style unlabeled examples
- invalid JSON top-level shapes
- graph selection by index and id

## Notes

Evidence ids are assigned during evidence construction, not during loading. That keeps this stage focused on faithfully representing FinQA records.

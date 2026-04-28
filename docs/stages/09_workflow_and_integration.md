# 09. Workflow And Integration

## Purpose

Connect the individual stages into a simple end-to-end FinQA pipeline.

Detailed YAML options and run commands are documented in [Workflow And Run Configs](../details/workflow_and_configs.md).

## Architecture

The project uses two layers:

- a LangGraph workflow for one example
- a normal Python batch runner for many examples

This keeps the graph focused and keeps dataset-level concerns, output files, and evaluation in regular Python code.

## Single-Example Workflow

Implemented in `src/graph/workflow.py`:

```text
selected_example
-> add_question
-> build_evidence
-> retrieve_evidence
-> build_prompt
-> call_model
-> parse_model_output
-> execute_parsed_output
-> format_prediction
-> prediction
```

The public helper is:

```python
run_single_example_workflow(...)
```

It returns one `Prediction`.

## Batch Runner

Implemented in `scripts/run_pipeline.py`.

The runner:

- reads a YAML config
- loads and slices examples
- builds retrieval and model configs
- selects the model client
- runs the single-example workflow for each example
- writes predictions
- optionally evaluates predictions when gold labels exist
- can continue across ordinary example-level errors
- can stop early on provider-level errors

## Key Files

- `src/graph/workflow.py`
- `scripts/run_pipeline.py`
- `configs/runs/template.yaml`
- `tests/test_single_example_workflow.py`
- `tests/test_run_pipeline_script.py`

## Error Handling

`continue_on_error` is useful for ordinary per-example failures because it preserves a prediction record with errors.

`abort_on_provider_error` stops the batch when the configured provider is unavailable or blocked. This prevents a quota, auth, or rate-limit issue from producing a full predictions file of null answers.

## Verification

Tests cover:

- composed graph nodes
- single-example workflow behavior
- config loading
- provider-client selection
- prediction writing
- batch runner behavior
- OpenAI provider-error fail-fast behavior

## Possible Improvements

- Add run metadata such as git commit, prompt variant, and model settings to output files.
- Add a dry-run mode that builds prompts without calling a model.
- Add a small report-generation script that summarizes batch outputs.

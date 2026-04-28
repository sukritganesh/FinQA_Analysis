# Workflow And Run Configs

This document explains how the end-to-end pipeline is wired and how YAML run configs control batch execution.

For the high-level architecture, see [09. Workflow And Integration](../stages/09_workflow_and_integration.md).

## Execution Layers

The project has two execution layers:

- `src/graph/workflow.py`: LangGraph workflow for one `FinQAExample`
- `scripts/run_pipeline.py`: batch runner that loads examples, calls the workflow, writes predictions, and optionally evaluates results

The single-example workflow returns one `Prediction`. The batch runner manages files and dataset-level evaluation.

## Single-Example Workflow

The graph shape is:

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

## Batch Runner

Run a batch from the repository root:

```bash
.venv/bin/python scripts/run_pipeline.py configs/runs/train_100_qwen14b_awq.yaml
```

The runner:

- reads a YAML config
- loads examples
- applies `offset` and `limit`
- builds retrieval and model configs
- creates the provider client
- runs the graph once per example
- writes predictions
- optionally writes evaluation summaries

## Config Sections

### `data`

```yaml
data:
  input_path: data/raw/dev.json
  offset: 0
  limit: 30
```

- `input_path`: FinQA JSON file to load.
- `offset`: first example index to run.
- `limit`: number of examples to run. Omit it to run the rest of the split.

### `retrieval`

```yaml
retrieval:
  strategy: bm25
  mode: by_source
  top_k: 5
  top_k_text: 3
  top_k_table: 3
```

- `strategy`: currently `bm25`.
- `mode`: `combined` or `by_source`.
- `top_k`: selected evidence count for `combined` mode.
- `top_k_text`: selected text evidence count for `by_source` mode.
- `top_k_table`: selected table evidence count for `by_source` mode.

See [BM25 Retrieval](bm25_retrieval.md) for strategy details.

### `prompt`

```yaml
prompt:
  prompt_dir: configs/prompts/finqa_prompt_B_compact
```

`prompt_dir` points to a prompt asset folder. Prompt assets are loaded by `src/llm/prompts.py`.

### `model`

```yaml
model:
  provider: vllm
  model: qwen14b-awq
  base_url: http://127.0.0.1:8000/v1
  max_tokens: 256
  temperature: 0.0
  timeout_seconds: 60.0
```

`provider` can be `vllm` or `openai`. If omitted, the code defaults to `vllm`.

For OpenAI:

```yaml
model:
  provider: openai
  model: gpt-4.1-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  max_tokens: 256
  temperature: 0.0
  timeout_seconds: 60.0
```

See [Model Providers](model_providers.md) for provider setup, aliases, and serving commands.

### `outputs`

```yaml
outputs:
  run_name: dev_30_qwen14b_awq
  prediction_path: reports/test_results/dev_30_qwen14b_awq/predictions.json
  evaluation_dir: reports/test_results/dev_30_qwen14b_awq
  evaluate: true
```

- `run_name`: name used in evaluation output files.
- `prediction_path`: where compact predictions are written.
- `evaluation_dir`: where summary/details files are written.
- `evaluate`: whether to evaluate predictions against gold labels when available.

### `runtime`

```yaml
runtime:
  continue_on_error: true
  abort_on_provider_error: true
```

- `continue_on_error`: preserve failed per-example predictions instead of crashing the batch.
- `abort_on_provider_error`: stop early on provider-level failures such as invalid API keys, quota errors, or rate limits.

## Example Configs

- `configs/runs/template.yaml`
- `configs/runs/dev_30_qwen14b_awq.yaml`
- `configs/runs/train_100_qwen14b_awq.yaml`
- `configs/runs/train_100_openai.yaml`

## Common Runs

Local vLLM run:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --max-model-len 4096
.venv/bin/python scripts/run_pipeline.py configs/runs/train_100_qwen14b_awq.yaml
```

OpenAI run:

```bash
export OPENAI_API_KEY="sk-..."
.venv/bin/python scripts/test_openai_connection.py
.venv/bin/python scripts/run_pipeline.py configs/runs/train_100_openai.yaml
```

Retrieval-only evaluation:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/test.json --limit 100 --mode by_source --top-k-text 3 --top-k-table 3 --include-hits --log-path reports/retrieval/retrieval_eval_test_100_by_source_3_3_with_hits.md
```

## Output Files

Prediction file:

```text
reports/test_results/<run_name>/predictions.json
```

Evaluation files:

```text
reports/test_results/<run_name>/<run_name>_summary.json
reports/test_results/<run_name>/<run_name>_details.json
reports/test_results/<run_name>/<run_name>_details.md
```

These files are intended to support both automated scoring and manual error analysis.

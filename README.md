# FinQA Financial QA Pipeline

This repository implements an inference-first question-answering pipeline for the FinQA dataset. It answers financial numerical reasoning questions over report text and tables by retrieving evidence, prompting a model, parsing a constrained output, executing arithmetic in Python, and evaluating predictions against gold answers when available.

## Pipeline

```text
FinQA JSON
-> normalized examples
-> evidence units
-> BM25 retrieval
-> prompt generation
-> model invocation
-> deterministic execution
-> prediction formatting
-> evaluation
```

Runtime inputs are kept separate from gold labels so inference code does not rely on evaluation-only fields.

## Repository Layout

```text
configs/   Run configs and prompt assets
data/      Local FinQA JSON files
docs/      Stage notes and technical details
reports/   Generated predictions, evaluations, and notes
scripts/   CLI helpers and experiment runners
src/       Main Python package
tests/     Unit and integration tests
```

## Setup

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Put FinQA files under `data/raw/`:

```text
data/raw/train.json
data/raw/dev.json
data/raw/test.json
data/raw/private_test.json
```

## Run Tests

```bash
.venv/bin/python -m pytest -q
```

## Run The Pipeline With vLLM

Start vLLM in a separate shell (make sure vllm is up and running - you can use ChatGPT for this).

Make sure you have a graphics card and CUDA installed.

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --max-model-len 4096
```

Then run a batch config:

```bash
.venv/bin/python scripts/run_pipeline.py configs/runs/sandbox.yaml
```

Outputs are written under `reports/test_results/`.

## Run The Pipeline With OpenAI

Set your API key and test connectivity:

```bash
export OPENAI_API_KEY="sk-..."
.venv/bin/python scripts/test_openai_connection.py
```

Run the OpenAI config:

```bash
.venv/bin/python scripts/run_pipeline.py configs/runs/train_100_openai.yaml
```

If OpenAI returns `insufficient_quota`, the request reached OpenAI but the account or project cannot bill for API usage. The OpenAI config stops early on provider-level failures so a bad provider state does not produce a full batch of null predictions.

## Useful Commands

Inspect examples:

```bash
.venv/bin/python scripts/inspect_example.py --input data/raw/test.json --limit 1
```

Evaluate retrieval only:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/test.json --limit 100 --mode by_source --top-k-text 3 --top-k-table 3 --include-hits --log-path reports/retrieval/retrieval_eval_test_100_by_source_3_3_with_hits.md
```

Preview a prompt:

```bash
.venv/bin/python scripts/preview_prompt.py --input data/raw/test.json --index 0 --mode by_source --top-k-text 2 --top-k-table 2 --prompt-dir configs/prompts/finqa_prompt_A
```

## Documentation

Start here:

- [Workflow And Run Configs](docs/details/workflow_and_configs.md)
- [Model Providers](docs/details/model_providers.md)
- [FinQA Dataset](docs/details/finqa_dataset.md)
- [BM25 Retrieval](docs/details/bm25_retrieval.md)
- [Executable Program Contract](docs/details/executable_programs.md)

Stage notes:

- [01. Data Loading](docs/stages/01_data_loading.md)
- [02. Evidence Construction](docs/stages/02_evidence_construction.md)
- [03. Retrieval](docs/stages/03_retrieval.md)
- [04. Prompt Generation](docs/stages/04_prompt_generation.md)
- [05. Model Invocation](docs/stages/05_model_invocation.md)
- [06. Deterministic Execution](docs/stages/06_deterministic_execution.md)
- [07. Validation And Formatting](docs/stages/07_validation_and_formatting.md)
- [08. Evaluation](docs/stages/08_evaluation.md)
- [09. Workflow And Integration](docs/stages/09_workflow_and_integration.md)
- [10. Report And Presentation](docs/stages/10_report_and_presentation.md)

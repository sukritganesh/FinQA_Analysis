# Model Providers

This document describes the model-provider layer used by the FinQA pipeline.

The pipeline treats model invocation as a small interface:

```text
prompt -> provider client -> model_output_text
```

Parsing, execution, validation, and evaluation happen after the model returns text.

## Supported Providers

The current provider names are:

- `vllm`: local OpenAI-compatible serving
- `openai`: hosted OpenAI Chat Completions API

Both providers use the same `ModelConfig` and return the same `ModelResponse` shape.

## Key Files

- `src/llm/client.py`
- `src/graph/model_call.py`
- `scripts/test_openai_connection.py`
- `scripts/vllm/`
- `configs/runs/train_100_qwen14b_awq.yaml`
- `configs/runs/train_100_openai.yaml`

## ModelConfig

`ModelConfig` contains:

- `provider`
- `model`
- `base_url`
- `max_tokens`
- `temperature`
- `timeout_seconds`
- `extra_body`
- optional `api_key`
- optional `api_key_env`

Default local endpoint:

```text
http://127.0.0.1:8000/v1
```

Default OpenAI endpoint:

```text
https://api.openai.com/v1
```

## Local Model Aliases

Aliases are resolved before requests are sent. They are convenience names for configs and scripts.

| Alias | Model id |
| --- | --- |
| `qwen3b` | `Qwen/Qwen2.5-3B-Instruct` |
| `qwen7b-awq` | `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| `qwen14b-awq` | `Qwen/Qwen2.5-14B-Instruct-AWQ` |
| `deepseek-qwen1.5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| `deepseek-qwen7b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| `deepseek-qwen14b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` |

The served model id must match what vLLM exposes.

## vLLM Provider

Use vLLM for local Hugging Face model serving. The main pipeline assumes the server is already running.

Example:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --max-model-len 4096
```

Then run a pipeline config that points to the local endpoint:

```yaml
model:
  provider: vllm
  model: qwen14b-awq
  base_url: http://127.0.0.1:8000/v1
  max_tokens: 256
  temperature: 0.0
  timeout_seconds: 60.0
```

### Useful vLLM Commands

Qwen instruct:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --max-model-len 4096
```

7B AWQ:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --max-model-len 4096
```

14B AWQ:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --max-model-len 4096
```

DeepSeek distilled Qwen:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --max-model-len 4096
```

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max-model-len 4096
```

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --max-model-len 4096
```

## OpenAI Provider

Use OpenAI for hosted experiments with the same pipeline interface.

Example:

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

Prefer `api_key_env` over storing API keys in YAML files.

Before a batch run, test connectivity:

```bash
export OPENAI_API_KEY="sk-..."
.venv/bin/python scripts/test_openai_connection.py
```

It is NOT recommended to add API keys as permanent environmental variables, for security purposes.

Then run:

```bash
.venv/bin/python scripts/run_pipeline.py configs/runs/train_100_openai.yaml
```

If OpenAI returns `insufficient_quota`, the request reached OpenAI but the account or project cannot bill for API usage.

## Provider Error Handling

The graph-level model-call node appends errors instead of crashing.

The batch runner can stop early on provider-level failures when:

```yaml
runtime:
  abort_on_provider_error: true
```

This prevents a provider outage, invalid key, quota issue, or rate-limit problem from producing a full predictions file of null answers.

## Manual vLLM Request

When vLLM is running, this request should work:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [
      {"role": "user", "content": "Return exactly one line: subtract(10, 4)"}
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
```

## Notes

- Regular tests use fake or recording clients; they do not require live vLLM or OpenAI access.
- vLLM setup is environment-specific and may be easier on Linux or WSL than native Windows.
- Keep provider-specific options in `extra_body` unless they become common enough to promote into `ModelConfig`.

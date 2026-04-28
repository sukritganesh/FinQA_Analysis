# 05. Model Invocation

## Purpose

Send one assembled prompt to a configured model provider and capture the raw response text.

This stage is intentionally narrow:

```text
prompt -> model provider -> model_output_text
```

Parsing, execution, validation, and evaluation happen later.

Provider setup, model aliases, and serving commands are documented in [Model Providers](../details/model_providers.md).

## What This Stage Does

- Builds OpenAI-compatible chat-completion requests.
- Supports local vLLM and hosted OpenAI through one interface.
- Resolves configured model aliases.
- Captures response text and optional metadata.
- Records provider errors for the workflow and batch runner.

## Key Files

- `src/llm/client.py`
- `src/graph/model_call.py`
- `scripts/test_openai_connection.py`
- `scripts/vllm/`
- `tests/test_model_client.py`
- `tests/test_model_call_graph.py`

## Stage Inputs

- `prompt`
- `model_config`
- optional injected `model_client` for tests or custom providers

`ModelConfig` includes provider name, model name, base URL, generation settings, timeout, optional provider-specific request body fields, and optional API-key settings.

## Stage Outputs

- `model_response`
- `model_output_text`
- accumulated `errors`

The raw model text is not parsed here. The next stage decides whether the output is a direct answer, executable program, or malformed response.

## Provider Boundary

The provider clients live behind the `ModelClient` protocol:

```text
generate(prompt, config) -> ModelResponse
```

This keeps the rest of the pipeline independent from whether the prompt is sent to local vLLM, OpenAI, or a future provider.

## Error Handling

The model-call graph node appends errors instead of crashing. This keeps one-example graph execution inspectable.

For batch runs, provider-level failures can still stop the run early through `abort_on_provider_error`. That prevents a bad API key, quota problem, server outage, or rate-limit issue from producing a full file of null predictions.

## LangGraph Integration

The model-call node is separate from prompt generation:

```text
build_prompt -> call_model
```

It consumes `prompt`, `model_config`, and an optional `model_client`.

It does not:

- build prompts
- parse model output
- execute arithmetic
- evaluate answers

## Verification

Tests use fake or recording clients so normal pytest stays offline. They cover:

- model config defaults
- model aliases
- vLLM request payloads
- OpenAI request payloads
- placeholder or missing API key handling
- graph behavior for successful and failed model calls

## Possible Improvements

- Add retry/backoff for transient local server errors.
- Record latency in `ModelResponse`.
- Add new providers behind the existing `ModelClient` protocol.
- Promote provider-specific request options only when experiments need them repeatedly.

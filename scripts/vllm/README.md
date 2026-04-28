# vLLM Helper Scripts

This folder contains small helper scripts for manual vLLM checks. The main pipeline usually assumes vLLM is already running.

Shared provider setup and model options are documented in [Model Providers](../../docs/details/model_providers.md).

## One-Shot Helper

Start vLLM, send one prompt, print the response, and stop the server:

```bash
python scripts/vllm/run_qwen_once.py "Return exactly one line: subtract(10, 4)"
```

If your vLLM environment lives somewhere else:

```bash
python scripts/vllm/run_qwen_once.py "Hello there! How's it going?" --env-dir /path/to/.finqa-vllm-env
```

Notes:

- The helper looks for `<env-dir>/bin/activate` and `<env-dir>/Scripts/activate`.
- It uses the OpenAI-compatible vLLM HTTP API at `http://127.0.0.1:8000`.
- On Unix-like systems, it starts vLLM in a separate process group so interrupt cleanup can stop the served stack more reliably.

## Prompt File Helper

Send an existing prompt file to an already-running vLLM server:

```bash
python scripts/vllm/run_prompt_file.py --prompt-file reports/prompts/samples/testPrompt.txt --model Qwen/Qwen2.5-7B-Instruct-AWQ
```

Optionally write the model output:

```bash
python scripts/vllm/run_prompt_file.py --prompt-file reports/prompts/samples/testPrompt.txt --model Qwen/Qwen2.5-7B-Instruct-AWQ --output reports/prompts/test_example_0_output.txt
```

## Direct Serve Example

For normal pipeline runs, start vLLM yourself:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --max-model-len 4096
```

Then run a config that points at `http://127.0.0.1:8000/v1`.

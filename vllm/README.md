# vLLM Helper Scripts

This folder contains small manual checks for a local vLLM server. The main
pipeline assumes vLLM is already running and calls its OpenAI-compatible API at
`http://127.0.0.1:8000/v1`.

For the full model/provider list, see
[Model Providers](../docs/details/model_providers.md).

## Basic Flow

Use one terminal for vLLM and another terminal for pipeline/helper commands.

```bash
source .finqa-vllm-env/bin/activate
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --max-model-len 4096
```

Then, from the project environment:

```bash
source .venv/bin/activate
python vllm/run_prompt_file.py --prompt-file vllm/test_prompt.txt --model Qwen/Qwen2.5-14B-Instruct-AWQ
```

The model name passed to `--model` must match the model name served by vLLM,
unless vLLM was started with `--served-model-name`.

## Serving Commands

Small Qwen smoke test:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.85
```

14B AWQ model used for stronger local runs:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --max-model-len 4096 --gpu-memory-utilization 0.90
```

DeepSeek distilled Qwen 1.5B quick test:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --max-model-len 4096 --gpu-memory-utilization 0.80
```

Serve on a different port:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --host 127.0.0.1 --port 8001 --max-model-len 4096
```

Expose a shorter served model name:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --served-model-name qwen14b-awq --max-model-len 4096
```

## Prompt Helper

Send a prompt file to an already-running vLLM server:

```bash
python vllm/run_prompt_file.py --prompt-file vllm/test_prompt.txt --model Qwen/Qwen2.5-14B-Instruct-AWQ
```

Control generation length and sampling:

```bash
python vllm/run_prompt_file.py --prompt-file vllm/test_prompt.txt --model Qwen/Qwen2.5-14B-Instruct-AWQ --max-tokens 1024 --temperature 0.0
```

Call a server on a custom port and save the output:

```bash
python vllm/run_prompt_file.py --host 127.0.0.1 --port 8001 --prompt-file vllm/test_prompt.txt --model qwen14b-awq --max-tokens 512 --temperature 0.2 --output vllm/sandbox/test_output.txt
```

## Direct API Test

With vLLM running, you can paste a request directly into the terminal:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":"Answer with one word: ready?"}],"max_tokens":16,"temperature":0}'
```

If you used `--served-model-name qwen14b-awq`, use that model name in the
request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen14b-awq","messages":[{"role":"user","content":"Compute 2 + 2. Return only the number."}],"max_tokens":8,"temperature":0}'
```

## Hugging Face Downloads

Install or update the Hugging Face CLI:

```bash
pip install -U huggingface_hub
```

Log in if the model is gated or if you want authenticated downloads:

```bash
hf auth login
```

Older installs may use:

```bash
huggingface-cli login
```

Download a model into the local Hugging Face cache:

```bash
hf download Qwen/Qwen2.5-14B-Instruct-AWQ
```

Download to a specific folder:

```bash
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir models/deepseek-qwen1.5b
```

vLLM can serve either a Hugging Face model id or a local model directory.

## Common Issues

- **Connection refused**: vLLM is not running, is on another port, or is still
  loading the model.
- **Model name mismatch**: the API `model` field must match the served model id
  or the value passed to `--served-model-name`.
- **CUDA out of memory**: lower `--gpu-memory-utilization`, lower
  `--max-model-len`, use a smaller/AWQ model, or close other GPU processes.
- **Gated model or 401 errors**: run `hf auth login`, accept the model terms on
  Hugging Face if required, and retry.
- **Very slow first request**: first-token latency is usually higher while vLLM
  warms up kernels and fills caches.
- **Different output across runs**: use `--temperature 0.0` in helper requests
  and `temperature: 0.0` in pipeline configs for the most stable behavior.

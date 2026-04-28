#!/usr/bin/env python3
"""Send a prompt text file to an already-running vLLM server."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_PROMPT_FILE = "reports/prompts/test_example_0_prompt.txt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path(DEFAULT_PROMPT_FILE),
        help=f"Prompt file to send. Default: {DEFAULT_PROMPT_FILE}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Served vLLM model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="vLLM host. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM port. Default: 8000",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum output tokens. Default: 128",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the model output.",
    )
    return parser.parse_args()


def send_prompt(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    """Send one chat completion request to vLLM's OpenAI-compatible API."""
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM request failed with HTTP {exc.code}: {details}") from exc


def extract_text(response_json: dict) -> str:
    """Extract assistant text from an OpenAI-compatible chat response."""
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError(f"Response did not contain choices: {response_json}")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"Response did not contain text content: {response_json}")
    return content.strip()


def main() -> int:
    """Read the prompt file, call vLLM, and print the response."""
    args = parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8")
    base_url = f"http://{args.host}:{args.port}"

    response_json = send_prompt(
        base_url=base_url,
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    output = extract_text(response_json)
    print(output)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

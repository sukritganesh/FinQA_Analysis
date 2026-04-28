#!/usr/bin/env python3
"""Quick sandbox for validating OpenAI connection in this FinQA repo."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.client import ModelClientError, ModelConfig, OpenAIClient


def build_openai_client() -> OpenAIClient:
    config = ModelConfig(
        provider="openai",
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "256")),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
        timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60.0")),
        api_key_env="OPENAI_API_KEY",
    )
    return OpenAIClient(default_config=config)


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY is not set in your environment.")

    client = build_openai_client()
    prompt = (
        "You are a friendly assistant. Reply with one short line confirming that the OpenAI connection is working. Reply in english, arabic, and tamil."
    )

    try:
        response = client.generate(prompt)
    except ModelClientError as exc:
        raise SystemExit(f"OpenAI connectivity test failed: {exc}") from exc

    print("OpenAI connectivity test succeeded")
    print("---")
    print(f"Model: {response.model}")
    print(f"Finish reason: {response.finish_reason}")
    print(f"Response:\n{response.text}")
    if response.usage:
        print("---")
        print("Usage:")
        for key, value in response.usage.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

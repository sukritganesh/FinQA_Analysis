"""Model client interfaces for local inference."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

MODEL_ALIASES = {
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen7b-awq": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "qwen14b-awq": "Qwen/Qwen2.5-14B-Instruct-AWQ",
    "teichai-qwen3-4b-opus": "TeichAI/Qwen3-4B-Instruct-2507-Claude-Opus-3-Distill",
    "gemma-4-e2b-it": "google/gemma-4-E2B-it",
    "deepseek-qwen1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-qwen7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-qwen14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}

ProviderName = Literal["vllm", "openai"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for one local model-generation request."""

    provider: ProviderName = "vllm"
    model: str = DEFAULT_VLLM_MODEL
    base_url: str = DEFAULT_VLLM_BASE_URL
    max_tokens: int = 256
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    api_key: str | None = None
    api_key_env: str | None = None

    @property
    def resolved_model(self) -> str:
        """Return the exact served model id, resolving local aliases."""
        return MODEL_ALIASES.get(self.model, self.model)

    @property
    def chat_completions_url(self) -> str:
        """Return the OpenAI-compatible chat completions endpoint."""
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def resolved_api_key(self) -> str | None:
        """Return an API key from config or an environment variable."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return os.getenv("OPENAI_API_KEY")


@dataclass(slots=True)
class ModelResponse:
    """A minimal wrapper around one model response."""

    text: str
    raw: object | None = None
    model: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


class ModelClientError(RuntimeError):
    """Raised when a model request fails or returns an unexpected response."""


class ModelClient(Protocol):
    """Protocol for swappable model clients."""

    def generate(self, prompt: str, config: ModelConfig | None = None) -> ModelResponse:
        """Generate a response for a prompt."""


class PlaceholderModelClient:
    """A tiny stub for early local wiring and tests."""

    def generate(self, prompt: str, config: ModelConfig | None = None) -> ModelResponse:
        # TODO: Replace with a real local client, such as a vLLM-backed adapter.
        return ModelResponse(
            text='{"answer": null, "reasoning": "TODO", "operation": null, "operands": []}',
            raw={"prompt_preview": prompt[:120]},
            model=(config or ModelConfig()).resolved_model,
        )


class VLLMClient:
    """Client for vLLM's OpenAI-compatible chat completions API."""

    def __init__(self, default_config: ModelConfig | None = None) -> None:
        self.default_config = default_config or ModelConfig()

    def generate(self, prompt: str, config: ModelConfig | None = None) -> ModelResponse:
        """Send a prompt to vLLM and return the assistant response text."""
        request_config = config or self.default_config
        payload = _build_chat_completion_payload(prompt, request_config)
        response_json = self._post_json(
            url=request_config.chat_completions_url,
            payload=payload,
            timeout_seconds=request_config.timeout_seconds,
        )
        return _parse_chat_completion_response(response_json)

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """POST JSON to a vLLM endpoint and decode the JSON response."""
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise ModelClientError(f"vLLM request failed with HTTP {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise ModelClientError(f"Could not reach vLLM server: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ModelClientError("Timed out waiting for vLLM response.") from exc

        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ModelClientError(f"vLLM returned invalid JSON: {response_body[:200]}") from exc

        if not isinstance(decoded, dict):
            raise ModelClientError("vLLM returned a non-object JSON response.")
        return decoded


class OpenAIClient:
    """Client for OpenAI's Chat Completions API."""

    def __init__(self, default_config: ModelConfig | None = None) -> None:
        self.default_config = default_config or ModelConfig(
            provider="openai",
            model=DEFAULT_OPENAI_MODEL,
            base_url=DEFAULT_OPENAI_BASE_URL,
        )

    def generate(self, prompt: str, config: ModelConfig | None = None) -> ModelResponse:
        """Send a prompt to OpenAI and return the assistant response text."""
        request_config = config or self.default_config
        api_key = request_config.resolved_api_key
        if not api_key or api_key == "<Insert-API-key>":
            raise ModelClientError("OpenAI API key is missing or still set to the placeholder.")

        payload = _build_openai_chat_completion_payload(prompt, request_config)
        response_json = self._post_json(
            url=request_config.chat_completions_url,
            payload=payload,
            timeout_seconds=request_config.timeout_seconds,
            api_key=api_key,
        )
        return _parse_chat_completion_response(response_json)

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
        api_key: str,
    ) -> dict[str, Any]:
        """POST JSON to OpenAI and decode the JSON response."""
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url=url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise ModelClientError(f"OpenAI request failed with HTTP {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise ModelClientError(f"Could not reach OpenAI API: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ModelClientError("Timed out waiting for OpenAI response.") from exc

        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ModelClientError(f"OpenAI returned invalid JSON: {response_body[:200]}") from exc

        if not isinstance(decoded, dict):
            raise ModelClientError("OpenAI returned a non-object JSON response.")
        return decoded


def _build_chat_completion_payload(prompt: str, config: ModelConfig) -> dict[str, Any]:
    """Build the OpenAI-compatible chat completion request body."""
    payload: dict[str, Any] = {
        "model": config.resolved_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }
    payload.update(config.extra_body)
    return payload


def _build_openai_chat_completion_payload(prompt: str, config: ModelConfig) -> dict[str, Any]:
    """Build the OpenAI Chat Completions request body."""
    payload: dict[str, Any] = {
        "model": config.resolved_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": config.max_tokens,
        "temperature": config.temperature,
    }
    payload.update(config.extra_body)
    return payload


def _parse_chat_completion_response(response_json: dict[str, Any]) -> ModelResponse:
    """Extract assistant text from a chat completions response."""
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelClientError("Chat completion response did not include any choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ModelClientError("Chat completion response choice had an unexpected shape.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ModelClientError("Chat completion response choice did not include a message.")

    content = message.get("content")
    if not isinstance(content, str):
        raise ModelClientError("Chat completion response message did not include text content.")

    usage = response_json.get("usage")
    return ModelResponse(
        text=content.strip(),
        raw=response_json,
        model=response_json.get("model"),
        finish_reason=first_choice.get("finish_reason"),
        usage=usage if isinstance(usage, dict) else None,
    )

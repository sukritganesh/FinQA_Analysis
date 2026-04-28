from __future__ import annotations

import pytest

from src.llm.client import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_VLLM_BASE_URL,
    MODEL_ALIASES,
    ModelClientError,
    ModelConfig,
    OpenAIClient,
    VLLMClient,
)


def test_model_config_defaults_to_local_vllm_endpoint() -> None:
    config = ModelConfig()

    assert config.provider == "vllm"
    assert config.base_url == "http://127.0.0.1:8000/v1"
    assert config.chat_completions_url == f"{DEFAULT_VLLM_BASE_URL}/chat/completions"


def test_model_config_can_target_openai_endpoint() -> None:
    config = ModelConfig(
        provider="openai",
        model="gpt-4.1-mini",
        base_url=DEFAULT_OPENAI_BASE_URL,
        api_key="sk-test",
    )

    assert config.resolved_model == "gpt-4.1-mini"
    assert config.chat_completions_url == f"{DEFAULT_OPENAI_BASE_URL}/chat/completions"
    assert config.resolved_api_key == "sk-test"


def test_model_config_resolves_known_aliases() -> None:
    config = ModelConfig(model="qwen14b-awq")

    assert config.resolved_model == MODEL_ALIASES["qwen14b-awq"]


def test_model_config_resolves_deepseek_qwen_1_5b_alias() -> None:
    config = ModelConfig(model="deepseek-qwen1.5b")

    assert config.resolved_model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def test_vllm_client_posts_prompt_and_returns_assistant_text() -> None:
    client = RecordingVLLMClient(
        response_json={
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "choices": [
                {
                    "message": {"role": "assistant", "content": " subtract(10, 4)\n"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        }
    )
    config = ModelConfig(model="qwen7b-awq", max_tokens=64, temperature=0.0)

    response = client.generate("Return a program.", config)

    assert response.text == "subtract(10, 4)"
    assert response.model == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert response.finish_reason == "stop"
    assert response.usage == {"prompt_tokens": 12, "completion_tokens": 5}
    assert client.last_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert client.last_payload == {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "Return a program."}],
        "max_tokens": 64,
        "temperature": 0.0,
    }


def test_vllm_client_allows_extra_request_body_fields() -> None:
    client = RecordingVLLMClient(
        response_json={
            "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
        }
    )
    config = ModelConfig(extra_body={"top_p": 0.9})

    client.generate("Answer directly.", config)

    assert client.last_payload is not None
    assert client.last_payload["top_p"] == 0.9


def test_vllm_client_raises_for_missing_choices() -> None:
    client = RecordingVLLMClient(response_json={"choices": []})

    with pytest.raises(ModelClientError, match="choices"):
        client.generate("Return one line.")


def test_vllm_client_raises_for_missing_message_content() -> None:
    client = RecordingVLLMClient(response_json={"choices": [{"message": {}}]})

    with pytest.raises(ModelClientError, match="text content"):
        client.generate("Return one line.")


def test_openai_client_posts_prompt_and_returns_assistant_text() -> None:
    client = RecordingOpenAIClient(
        response_json={
            "model": "gpt-4.1-mini",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "divide(12, 3)"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
    )
    config = ModelConfig(
        provider="openai",
        model="gpt-4.1-mini",
        base_url=DEFAULT_OPENAI_BASE_URL,
        api_key="sk-test",
        max_tokens=64,
        temperature=0.0,
    )

    response = client.generate("Return a program.", config)

    assert response.text == "divide(12, 3)"
    assert response.model == "gpt-4.1-mini"
    assert response.finish_reason == "stop"
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 4}
    assert client.last_url == "https://api.openai.com/v1/chat/completions"
    assert client.last_api_key == "sk-test"
    assert client.last_payload == {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "Return a program."}],
        "max_completion_tokens": 64,
        "temperature": 0.0,
    }


def test_openai_client_rejects_placeholder_api_key() -> None:
    client = RecordingOpenAIClient(response_json={})
    config = ModelConfig(
        provider="openai",
        model="gpt-4.1-mini",
        base_url=DEFAULT_OPENAI_BASE_URL,
        api_key="<Insert-API-key>",
    )

    with pytest.raises(ModelClientError, match="API key"):
        client.generate("Return one line.", config)


class RecordingVLLMClient(VLLMClient):
    def __init__(self, response_json: dict) -> None:
        super().__init__()
        self.response_json = response_json
        self.last_url: str | None = None
        self.last_payload: dict | None = None
        self.last_timeout_seconds: float | None = None

    def _post_json(
        self,
        url: str,
        payload: dict,
        timeout_seconds: float,
    ) -> dict:
        self.last_url = url
        self.last_payload = payload
        self.last_timeout_seconds = timeout_seconds
        return self.response_json


class RecordingOpenAIClient(OpenAIClient):
    def __init__(self, response_json: dict) -> None:
        super().__init__()
        self.response_json = response_json
        self.last_url: str | None = None
        self.last_payload: dict | None = None
        self.last_timeout_seconds: float | None = None
        self.last_api_key: str | None = None

    def _post_json(
        self,
        url: str,
        payload: dict,
        timeout_seconds: float,
        api_key: str,
    ) -> dict:
        self.last_url = url
        self.last_payload = payload
        self.last_timeout_seconds = timeout_seconds
        self.last_api_key = api_key
        return self.response_json

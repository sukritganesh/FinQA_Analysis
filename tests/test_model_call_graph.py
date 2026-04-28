from __future__ import annotations

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.graph.model_call import build_model_call_graph, call_model_node
from src.graph.prompting import build_prompt_node
from src.llm.client import ModelClientError, ModelConfig, ModelResponse
from src.retrieval.base import RetrievedEvidence
from src.data.schemas import EvidenceUnit


class PromptModelState(TypedDict):
    question: NotRequired[str]
    retrieved_evidence: NotRequired[list[RetrievedEvidence]]
    prompt: NotRequired[str]
    model_config: NotRequired[ModelConfig]
    model_client: NotRequired[FakeModelClient]
    model_response: NotRequired[ModelResponse]
    model_output_text: NotRequired[str]
    errors: NotRequired[list[str]]


def test_model_call_graph_writes_model_output_text() -> None:
    graph = build_model_call_graph()
    client = FakeModelClient(ModelResponse(text="subtract(10, 4)", model="fake-model"))

    result = graph.invoke(
        {
            "prompt": "Return exactly one line.",
            "model_config": ModelConfig(model="qwen7b-awq"),
            "model_client": client,
        }
    )

    assert result["errors"] == []
    assert result["model_output_text"] == "subtract(10, 4)"
    assert result["model_response"].model == "fake-model"
    assert client.seen_prompt == "Return exactly one line."
    assert client.seen_config is not None
    assert client.seen_config.resolved_model == "Qwen/Qwen2.5-7B-Instruct-AWQ"


def test_model_call_graph_records_missing_prompt_error() -> None:
    graph = build_model_call_graph()
    result = graph.invoke({"errors": []})

    assert "Missing prompt" in result["errors"][0]


def test_model_call_graph_records_client_error() -> None:
    graph = build_model_call_graph()
    client = FailingModelClient()

    result = graph.invoke({"prompt": "Hello", "model_client": client, "errors": []})

    assert "Failed to call model" in result["errors"][0]
    assert "server unavailable" in result["errors"][0]


def test_composed_prompt_and_model_call_graph() -> None:
    graph = _build_prompt_model_graph()
    evidence = EvidenceUnit(
        evidence_id="table_0",
        source="table",
        text="year the 2024 revenue of amount is 100 ;",
    )
    client = EchoingModelClient()

    result = graph.invoke(
        {
            "question": "What was revenue in 2024?",
            "retrieved_evidence": [
                RetrievedEvidence(unit=evidence, score=1.0, rank=1, selected=True)
            ],
            "model_client": client,
        }
    )

    assert result["errors"] == []
    assert "What was revenue in 2024?" in result["prompt"]
    assert result["model_output_text"] == "saw prompt"
    assert client.seen_prompt is not None
    assert "[table_0] year the 2024 revenue of amount is 100 ;" in client.seen_prompt


def _build_prompt_model_graph():
    graph = StateGraph(PromptModelState)
    graph.add_node("build_prompt", build_prompt_node)
    graph.add_node("call_model", call_model_node)
    graph.add_edge(START, "build_prompt")
    graph.add_edge("build_prompt", "call_model")
    graph.add_edge("call_model", END)
    return graph.compile()


class FakeModelClient:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.seen_prompt: str | None = None
        self.seen_config: ModelConfig | None = None

    def generate(self, prompt: str, config: ModelConfig | None = None) -> ModelResponse:
        self.seen_prompt = prompt
        self.seen_config = config
        return self.response


class EchoingModelClient(FakeModelClient):
    def __init__(self) -> None:
        super().__init__(ModelResponse(text="saw prompt"))


class FailingModelClient:
    def generate(self, prompt: str, config: ModelConfig | None = None) -> ModelResponse:
        raise ModelClientError("server unavailable")

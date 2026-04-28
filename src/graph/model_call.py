"""LangGraph integration for calling a local language model."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.llm.client import ModelClient, ModelConfig, ModelResponse, VLLMClient


class ModelCallState(TypedDict):
    """State for sending one prompt to one configured model client."""

    prompt: NotRequired[str]
    model_config: NotRequired[ModelConfig]
    model_client: NotRequired[ModelClient]
    model_response: NotRequired[ModelResponse]
    model_output_text: NotRequired[str]
    errors: NotRequired[list[str]]


def call_model_node(state: ModelCallState) -> ModelCallState:
    """Call the configured model client with the prompt in state."""
    prompt = state.get("prompt")
    if not prompt:
        return _append_error(state, "Missing prompt for model invocation.")

    config = state.get("model_config", ModelConfig())
    client = state.get("model_client") or VLLMClient(default_config=config)

    try:
        response = client.generate(prompt, config)
    except Exception as exc:  # noqa: BLE001
        return _append_error(state, f"Failed to call model: {exc}")

    return {
        **state,
        "model_response": response,
        "model_output_text": response.text,
        "errors": state.get("errors", []),
    }


def build_model_call_graph():
    """Build the minimal LangGraph model-invocation workflow."""
    graph = StateGraph(ModelCallState)
    graph.add_node("call_model", call_model_node)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_model", END)
    return graph.compile()


def _append_error(state: ModelCallState, message: str) -> ModelCallState:
    errors = [*state.get("errors", []), message]
    return {**state, "errors": errors}

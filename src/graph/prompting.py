"""LangGraph integration for FinQA prompt construction."""

from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.data.schemas import FinQAExample
from src.llm.prompts import DEFAULT_PROMPT_DIR, build_reasoning_prompt
from src.retrieval.base import RetrievedEvidence


class PromptGenerationState(TypedDict):
    """State for turning selected evidence into one model-ready prompt."""

    selected_example: NotRequired[FinQAExample]
    question: NotRequired[str]
    retrieved_evidence: NotRequired[list[RetrievedEvidence]]
    prompt_dir: NotRequired[str | Path]
    prompt: NotRequired[str]
    errors: NotRequired[list[str]]


def build_prompt_node(state: PromptGenerationState) -> PromptGenerationState:
    """Build the reasoning prompt for one selected FinQA example.

    This node intentionally stops before model inference. Keeping prompt
    generation separate makes the workflow easier to inspect and debug.
    """
    question = _get_question(state)
    if not question:
        return _append_error(state, "Missing question for prompt construction.")

    retrieved_evidence = state.get("retrieved_evidence")
    if retrieved_evidence is None:
        return _append_error(state, "Missing retrieved_evidence for prompt construction.")

    prompt_dir = state.get("prompt_dir", DEFAULT_PROMPT_DIR)

    try:
        prompt = build_reasoning_prompt(
            question=question,
            selected_evidence=retrieved_evidence,
            prompt_dir=prompt_dir,
        )
    except Exception as exc:  # noqa: BLE001
        return _append_error(state, f"Failed to build prompt: {exc}")

    return {
        **state,
        "question": question,
        "prompt": prompt,
        "errors": state.get("errors", []),
    }


def build_prompt_generation_graph():
    """Build the minimal LangGraph prompt-generation workflow."""
    graph = StateGraph(PromptGenerationState)
    graph.add_node("build_prompt", build_prompt_node)
    graph.add_edge(START, "build_prompt")
    graph.add_edge("build_prompt", END)
    return graph.compile()


def _get_question(state: PromptGenerationState) -> str | None:
    """Read the question from state or from the selected example."""
    question = state.get("question")
    if question:
        return question

    example = state.get("selected_example")
    if example is None:
        return None
    return example.runtime.question


def _append_error(state: PromptGenerationState, message: str) -> PromptGenerationState:
    errors = [*state.get("errors", []), message]
    return {**state, "errors": errors}

"""LangGraph integration for the FinQA data-loading stage."""

from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.data.loader import load_finqa_examples
from src.data.schemas import FinQAExample


class DataLoadingState(TypedDict):
    """State for loading a split and selecting one normalized example."""

    data_path: str
    example_index: NotRequired[int]
    example_id: NotRequired[str]
    examples: NotRequired[list[FinQAExample]]
    selected_example: NotRequired[FinQAExample]
    errors: NotRequired[list[str]]


def load_examples_node(state: DataLoadingState) -> DataLoadingState:
    """Load normalized examples from the path stored in graph state."""
    data_path = state.get("data_path")
    if not data_path:
        return _append_error(state, "Missing required data_path.")

    try:
        examples = load_finqa_examples(Path(data_path))
    except Exception as exc:  # noqa: BLE001
        return _append_error(state, f"Failed to load examples: {exc}")

    return {**state, "examples": examples, "errors": state.get("errors", [])}


def select_example_node(state: DataLoadingState) -> DataLoadingState:
    """Select one example by id or index after loading a split."""
    examples = state.get("examples", [])
    if not examples:
        return _append_error(state, "No examples are available to select.")

    example_id = state.get("example_id")
    if example_id:
        for example in examples:
            if example.runtime.example_id == example_id:
                return {**state, "selected_example": example, "errors": state.get("errors", [])}
        return _append_error(state, f"Example id not found: {example_id}")

    example_index = state.get("example_index", 0)
    if example_index < 0 or example_index >= len(examples):
        return _append_error(
            state,
            f"Example index {example_index} is out of range for {len(examples)} examples.",
        )

    return {**state, "selected_example": examples[example_index], "errors": state.get("errors", [])}


def build_data_loading_graph():
    """Build the minimal LangGraph data-loading workflow."""
    graph = StateGraph(DataLoadingState)
    graph.add_node("load_examples", load_examples_node)
    graph.add_node("select_example", select_example_node)
    graph.add_edge(START, "load_examples")
    graph.add_edge("load_examples", "select_example")
    graph.add_edge("select_example", END)
    return graph.compile()


def _append_error(state: DataLoadingState, message: str) -> DataLoadingState:
    errors = [*state.get("errors", []), message]
    return {**state, "errors": errors}

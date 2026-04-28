"""LangGraph integration for parsing and deterministic execution."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.data.schemas import FinQAExample
from src.llm.parser import ParsedReasoningOutput, parse_reasoning_output
from src.tools.executor import ExecutionResult, execute_parsed_output


class DeterministicExecutionState(TypedDict):
    """State for parsing model output and executing it deterministically."""

    selected_example: NotRequired[FinQAExample]
    model_output_text: NotRequired[str]
    parsed_output: NotRequired[ParsedReasoningOutput]
    execution_result: NotRequired[ExecutionResult]
    final_answer: NotRequired[str]
    errors: NotRequired[list[str]]


def parse_model_output_node(state: DeterministicExecutionState) -> DeterministicExecutionState:
    """Parse raw model output into a direct answer or linear program."""
    model_output_text = state.get("model_output_text")
    if not model_output_text:
        return _append_error(state, "Missing model_output_text for parsing.")

    try:
        parsed_output = parse_reasoning_output(model_output_text)
    except Exception as exc:  # noqa: BLE001
        return _append_error(state, f"Failed to parse model output: {exc}")

    return {
        **state,
        "parsed_output": parsed_output,
        "errors": state.get("errors", []),
    }


def execute_parsed_output_node(state: DeterministicExecutionState) -> DeterministicExecutionState:
    """Execute parsed output and write the final answer into graph state."""
    parsed_output = state.get("parsed_output")
    if parsed_output is None:
        return _append_error(state, "Missing parsed_output for deterministic execution.")

    table = _get_table_from_state(state)

    try:
        execution_result = execute_parsed_output(parsed_output, table=table)
    except Exception as exc:  # noqa: BLE001
        return _append_error(state, f"Failed to execute parsed output: {exc}")

    return {
        **state,
        "execution_result": execution_result,
        "final_answer": execution_result.final_answer,
        "errors": state.get("errors", []),
    }


def build_deterministic_execution_graph():
    """Build the minimal parsing plus deterministic-execution workflow."""
    graph = StateGraph(DeterministicExecutionState)
    graph.add_node("parse_model_output", parse_model_output_node)
    graph.add_node("execute_parsed_output", execute_parsed_output_node)
    graph.add_edge(START, "parse_model_output")
    graph.add_edge("parse_model_output", "execute_parsed_output")
    graph.add_edge("execute_parsed_output", END)
    return graph.compile()


def _get_table_from_state(state: DeterministicExecutionState) -> list[list[str]] | None:
    example = state.get("selected_example")
    if example is None:
        return None
    return example.runtime.table


def _append_error(
    state: DeterministicExecutionState,
    message: str,
) -> DeterministicExecutionState:
    errors = [*state.get("errors", []), message]
    return {**state, "errors": errors}

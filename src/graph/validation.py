"""LangGraph integration for final prediction formatting."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.data.schemas import FinQAExample
from src.eval.prediction import Prediction, build_prediction
from src.llm.parser import ParsedReasoningOutput
from src.retrieval.base import RetrievedEvidence
from src.tools.executor import ExecutionResult


class ValidationFormattingState(TypedDict):
    """State for packaging pipeline artifacts into a prediction object."""

    selected_example: NotRequired[FinQAExample]
    model_output_text: NotRequired[str]
    parsed_output: NotRequired[ParsedReasoningOutput]
    execution_result: NotRequired[ExecutionResult]
    retrieved_evidence: NotRequired[list[RetrievedEvidence]]
    final_answer: NotRequired[str]
    prediction: NotRequired[Prediction]
    errors: NotRequired[list[str]]


def format_prediction_node(state: ValidationFormattingState) -> ValidationFormattingState:
    """Build an evaluation-ready prediction from current graph state."""
    errors = list(state.get("errors", []))
    if "final_answer" not in state:
        errors.append("Missing final_answer for prediction formatting.")

    prediction = build_prediction(
        selected_example=state.get("selected_example"),
        final_answer=state.get("final_answer"),
        model_output_text=state.get("model_output_text"),
        parsed_output=state.get("parsed_output"),
        execution_result=state.get("execution_result"),
        selected_evidence=state.get("retrieved_evidence"),
        errors=errors,
    )

    return {**state, "prediction": prediction, "errors": errors}


def build_validation_formatting_graph():
    """Build the minimal final formatting workflow."""
    graph = StateGraph(ValidationFormattingState)
    graph.add_node("format_prediction", format_prediction_node)
    graph.add_edge(START, "format_prediction")
    graph.add_edge("format_prediction", END)
    return graph.compile()

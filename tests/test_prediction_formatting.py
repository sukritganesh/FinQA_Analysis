from __future__ import annotations

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.data.schemas import EvidenceUnit, FinQAExample, RuntimeInputs
from src.eval.prediction import Prediction, build_prediction
from src.graph.execution import execute_parsed_output_node, parse_model_output_node
from src.graph.validation import build_validation_formatting_graph, format_prediction_node
from src.llm.parser import ParsedReasoningOutput
from src.retrieval.base import RetrievedEvidence
from src.tools.executor import ExecutionResult


class ExecutionFormattingState(TypedDict):
    selected_example: NotRequired[FinQAExample]
    model_output_text: NotRequired[str]
    parsed_output: NotRequired[ParsedReasoningOutput]
    execution_result: NotRequired[ExecutionResult]
    final_answer: NotRequired[str]
    prediction: NotRequired[Prediction]
    errors: NotRequired[list[str]]


def test_build_prediction_packages_pipeline_outputs() -> None:
    example = _example()
    parsed_state = parse_model_output_node({"model_output_text": "subtract(10, 4)"})
    executed_state = execute_parsed_output_node(parsed_state)

    prediction = build_prediction(
        selected_example=example,
        final_answer=executed_state["final_answer"],
        model_output_text="subtract(10, 4)",
        parsed_output=executed_state["parsed_output"],
        execution_result=executed_state["execution_result"],
        errors=[],
    )

    assert prediction.example_id == "example-1"
    assert prediction.question == "What is the result?"
    assert prediction.answer == "6"
    assert prediction.normalized_answer == "6"
    assert prediction.model_output_text == "subtract(10, 4)"
    assert prediction.parsed_output is executed_state["parsed_output"]
    assert prediction.execution_result is executed_state["execution_result"]
    assert prediction.selected_evidence == {}
    assert prediction.is_successful


def test_build_prediction_preserves_selected_evidence() -> None:
    evidence = RetrievedEvidence(
        unit=EvidenceUnit(evidence_id="table_1", source="table", text="the 2024 revenue is 10 ;"),
        score=1.5,
        rank=1,
        source_rank=1,
        selected=True,
    )

    prediction = build_prediction(
        selected_example=_example(),
        final_answer="10",
        model_output_text="10",
        parsed_output=None,
        execution_result=None,
        selected_evidence=[evidence],
        errors=[],
    )

    assert prediction.selected_evidence == {"table_1": "the 2024 revenue is 10 ;"}


def test_format_prediction_graph_writes_prediction_to_state() -> None:
    graph = build_validation_formatting_graph()
    result = graph.invoke(
        {
            "selected_example": _example(),
            "model_output_text": "yes",
            "final_answer": "YES",
            "errors": [],
        }
    )

    prediction = result["prediction"]
    assert result["errors"] == []
    assert prediction.answer == "YES"
    assert prediction.normalized_answer == "yes"
    assert prediction.example_id == "example-1"


def test_format_prediction_node_records_missing_answer_but_preserves_artifacts() -> None:
    result = format_prediction_node(
        {
            "selected_example": _example(),
            "model_output_text": "The answer is 6.",
            "errors": ["Failed to parse model output."],
        }
    )

    prediction = result["prediction"]
    assert "Missing final_answer" in result["errors"][-1]
    assert prediction.answer is None
    assert prediction.model_output_text == "The answer is 6."
    assert prediction.errors == tuple(result["errors"])
    assert not prediction.is_successful


def test_composed_parse_execute_and_format_graph() -> None:
    graph = _build_execution_formatting_graph()

    result = graph.invoke(
        {
            "selected_example": _example(),
            "model_output_text": "subtract(5829, 5735)",
            "errors": [],
        }
    )

    prediction = result["prediction"]
    assert result["errors"] == []
    assert result["final_answer"] == "94"
    assert prediction.answer == "94"
    assert prediction.parsed_output.kind == "program"
    assert prediction.execution_result.step_results[0].operation == "subtract"


def _build_execution_formatting_graph():
    graph = StateGraph(ExecutionFormattingState)
    graph.add_node("parse_model_output", parse_model_output_node)
    graph.add_node("execute_parsed_output", execute_parsed_output_node)
    graph.add_node("format_prediction", format_prediction_node)
    graph.add_edge(START, "parse_model_output")
    graph.add_edge("parse_model_output", "execute_parsed_output")
    graph.add_edge("execute_parsed_output", "format_prediction")
    graph.add_edge("format_prediction", END)
    return graph.compile()


def _example() -> FinQAExample:
    return FinQAExample(
        runtime=RuntimeInputs(
            example_id="example-1",
            filename=None,
            question="What is the result?",
            pre_text=[],
            post_text=[],
            table=[["Metric", "2021"], ["total", "10"]],
        )
    )

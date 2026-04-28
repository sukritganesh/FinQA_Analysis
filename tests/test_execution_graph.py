from __future__ import annotations

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.data.schemas import FinQAExample, RuntimeInputs
from src.graph.execution import (
    build_deterministic_execution_graph,
    execute_parsed_output_node,
    parse_model_output_node,
)
from src.graph.model_call import call_model_node
from src.llm.client import ModelConfig, ModelResponse
from src.llm.parser import ParsedReasoningOutput
from src.tools.executor import ExecutionResult


class ModelExecutionState(TypedDict):
    prompt: NotRequired[str]
    model_client: NotRequired[FakeModelClient]
    model_config: NotRequired[ModelConfig]
    model_output_text: NotRequired[str]
    model_response: NotRequired[ModelResponse]
    selected_example: NotRequired[FinQAExample]
    parsed_output: NotRequired[ParsedReasoningOutput]
    execution_result: NotRequired[ExecutionResult]
    final_answer: NotRequired[str]
    errors: NotRequired[list[str]]


def test_deterministic_execution_graph_parses_and_executes_program() -> None:
    graph = build_deterministic_execution_graph()

    result = graph.invoke({"model_output_text": "subtract(5829, 5735)"})

    assert result["errors"] == []
    assert result["parsed_output"].kind == "program"
    assert result["final_answer"] == "94"
    assert result["execution_result"].step_results[0].operation == "subtract"


def test_deterministic_execution_graph_executes_table_operation_from_selected_example() -> None:
    graph = build_deterministic_execution_graph()
    example = _example_with_table(
        [
            ["Metric", "2021", "2022", "2023"],
            ["total obligations", "10", "20", "30"],
        ]
    )

    result = graph.invoke(
        {
            "selected_example": example,
            "model_output_text": "table_sum(total obligations, none)",
        }
    )

    assert result["errors"] == []
    assert result["final_answer"] == "60"


def test_parse_model_output_node_records_parse_error() -> None:
    result = parse_model_output_node({"model_output_text": "The answer is 94.", "errors": []})

    assert "Failed to parse model output" in result["errors"][0]


def test_execute_parsed_output_node_records_execution_error() -> None:
    parsed_state = parse_model_output_node(
        {"model_output_text": "table_sum(total obligations, none)", "errors": []}
    )

    result = execute_parsed_output_node(parsed_state)

    assert "Failed to execute parsed output" in result["errors"][0]
    assert "requires table data" in result["errors"][0]


def test_composed_model_call_parse_and_execute_graph() -> None:
    graph = _build_model_execution_graph()
    client = FakeModelClient(ModelResponse(text="subtract(10, 4)"))

    result = graph.invoke({"prompt": "Return exactly one line.", "model_client": client})

    assert result["errors"] == []
    assert result["model_output_text"] == "subtract(10, 4)"
    assert result["parsed_output"].kind == "program"
    assert result["final_answer"] == "6"


def _build_model_execution_graph():
    graph = StateGraph(ModelExecutionState)
    graph.add_node("call_model", call_model_node)
    graph.add_node("parse_model_output", parse_model_output_node)
    graph.add_node("execute_parsed_output", execute_parsed_output_node)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_model", "parse_model_output")
    graph.add_edge("parse_model_output", "execute_parsed_output")
    graph.add_edge("execute_parsed_output", END)
    return graph.compile()


def _example_with_table(table: list[list[str]]) -> FinQAExample:
    return FinQAExample(
        runtime=RuntimeInputs(
            example_id="example-1",
            filename=None,
            question="What is the total?",
            pre_text=[],
            post_text=[],
            table=table,
        )
    )


class FakeModelClient:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response

    def generate(self, prompt: str, config: ModelConfig | None = None) -> ModelResponse:
        return self.response

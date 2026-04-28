"""End-to-end single-example FinQA inference workflow."""

from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.data.schemas import EvidenceUnit, FinQAExample
from src.eval.prediction import Prediction
from src.graph.evidence import build_evidence_node
from src.graph.execution import execute_parsed_output_node, parse_model_output_node
from src.graph.model_call import call_model_node
from src.graph.prompting import build_prompt_node
from src.graph.retrieval import retrieve_evidence_node
from src.graph.validation import format_prediction_node
from src.llm.client import ModelClient, ModelConfig, ModelResponse
from src.llm.parser import ParsedReasoningOutput
from src.retrieval.base import RetrievedEvidence, RetrievalConfig, RetrievalResult
from src.tools.executor import ExecutionResult


class SingleExampleWorkflowState(TypedDict):
    """State for one full FinQA inference pass."""

    selected_example: FinQAExample
    question: NotRequired[str]
    evidence_units: NotRequired[list[EvidenceUnit]]
    retrieval_config: NotRequired[RetrievalConfig]
    retrieval_result: NotRequired[RetrievalResult]
    ranked_evidence: NotRequired[list[RetrievedEvidence]]
    retrieved_evidence: NotRequired[list[RetrievedEvidence]]
    prompt_dir: NotRequired[str | Path]
    prompt: NotRequired[str]
    model_config: NotRequired[ModelConfig]
    model_client: NotRequired[ModelClient]
    model_response: NotRequired[ModelResponse]
    model_output_text: NotRequired[str]
    parsed_output: NotRequired[ParsedReasoningOutput]
    execution_result: NotRequired[ExecutionResult]
    final_answer: NotRequired[str]
    prediction: NotRequired[Prediction]
    errors: NotRequired[list[str]]


def add_question_node(state: SingleExampleWorkflowState) -> SingleExampleWorkflowState:
    """Copy the selected example question into graph state."""
    example = state.get("selected_example")
    if example is None:
        return _append_error(state, "Missing selected_example for workflow.")
    return {
        **state,
        "question": example.runtime.question,
        "errors": state.get("errors", []),
    }


def build_single_example_workflow():
    """Build the full single-example LangGraph workflow."""
    graph = StateGraph(SingleExampleWorkflowState)
    graph.add_node("add_question", add_question_node)
    graph.add_node("build_evidence", build_evidence_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("build_prompt", build_prompt_node)
    graph.add_node("call_model", call_model_node)
    graph.add_node("parse_model_output", parse_model_output_node)
    graph.add_node("execute_parsed_output", execute_parsed_output_node)
    graph.add_node("format_prediction", format_prediction_node)

    graph.add_edge(START, "add_question")
    graph.add_edge("add_question", "build_evidence")
    graph.add_edge("build_evidence", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "build_prompt")
    graph.add_edge("build_prompt", "call_model")
    graph.add_edge("call_model", "parse_model_output")
    graph.add_edge("parse_model_output", "execute_parsed_output")
    graph.add_edge("execute_parsed_output", "format_prediction")
    graph.add_edge("format_prediction", END)
    return graph.compile()


def run_single_example_workflow(
    example: FinQAExample,
    *,
    retrieval_config: RetrievalConfig | None = None,
    prompt_dir: str | Path | None = None,
    model_config: ModelConfig | None = None,
    model_client: ModelClient | None = None,
) -> Prediction:
    """Run one FinQA example through the full inference workflow."""
    graph = build_single_example_workflow()
    initial_state: SingleExampleWorkflowState = {
        "selected_example": example,
        "errors": [],
    }
    if retrieval_config is not None:
        initial_state["retrieval_config"] = retrieval_config
    if prompt_dir is not None:
        initial_state["prompt_dir"] = prompt_dir
    if model_config is not None:
        initial_state["model_config"] = model_config
    if model_client is not None:
        initial_state["model_client"] = model_client

    result = graph.invoke(initial_state)
    return result["prediction"]


def _append_error(
    state: SingleExampleWorkflowState,
    message: str,
) -> SingleExampleWorkflowState:
    errors = [*state.get("errors", []), message]
    return {**state, "errors": errors}

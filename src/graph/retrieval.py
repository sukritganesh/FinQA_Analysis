"""LangGraph integration for the FinQA retrieval stage."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.data.schemas import EvidenceUnit
from src.retrieval.base import RetrievedEvidence, RetrievalConfig, RetrievalResult
from src.retrieval.factory import build_retriever


class RetrievalState(TypedDict):
    """State for scoring and selecting evidence for one question."""

    question: NotRequired[str]
    evidence_units: NotRequired[list[EvidenceUnit]]
    retrieval_config: NotRequired[RetrievalConfig]
    retrieval_result: NotRequired[RetrievalResult]
    ranked_evidence: NotRequired[list[RetrievedEvidence]]
    retrieved_evidence: NotRequired[list[RetrievedEvidence]]
    errors: NotRequired[list[str]]


def retrieve_evidence_node(state: RetrievalState) -> RetrievalState:
    """Score all evidence and select top-k retrieved evidence."""
    question = state.get("question")
    if not question:
        return _append_error(state, "Missing question for retrieval.")

    evidence_units = state.get("evidence_units")
    if evidence_units is None:
        return _append_error(state, "Missing evidence_units for retrieval.")

    config = state.get("retrieval_config", RetrievalConfig())

    try:
        retriever = build_retriever(config.strategy)
        result = retriever.retrieve(question, evidence_units, config)
    except Exception as exc:  # noqa: BLE001
        return _append_error(state, f"Failed to retrieve evidence: {exc}")

    return {
        **state,
        "retrieval_result": result,
        "ranked_evidence": result.ranked_evidence,
        "retrieved_evidence": result.selected_evidence,
        "errors": state.get("errors", []),
    }


def build_retrieval_graph():
    """Build the minimal LangGraph retrieval workflow."""
    graph = StateGraph(RetrievalState)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_edge(START, "retrieve_evidence")
    graph.add_edge("retrieve_evidence", END)
    return graph.compile()


def _append_error(state: RetrievalState, message: str) -> RetrievalState:
    errors = [*state.get("errors", []), message]
    return {**state, "errors": errors}

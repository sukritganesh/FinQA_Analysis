"""LangGraph integration for the FinQA evidence-construction stage."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.data.evidence import build_evidence_units
from src.data.schemas import EvidenceUnit, FinQAExample


class EvidenceConstructionState(TypedDict):
    """State for constructing evidence for one selected example."""

    selected_example: NotRequired[FinQAExample]
    evidence_units: NotRequired[list[EvidenceUnit]]
    errors: NotRequired[list[str]]


def build_evidence_node(state: EvidenceConstructionState) -> EvidenceConstructionState:
    """Build retrieval-ready evidence units for one selected example."""
    example = state.get("selected_example")
    if example is None:
        return _append_error(state, "Missing selected_example for evidence construction.")

    evidence_units = build_evidence_units(example)
    return {
        **state,
        "evidence_units": evidence_units,
        "errors": state.get("errors", []),
    }


def build_evidence_construction_graph():
    """Build the minimal LangGraph evidence-construction workflow."""
    graph = StateGraph(EvidenceConstructionState)
    graph.add_node("build_evidence", build_evidence_node)
    graph.add_edge(START, "build_evidence")
    graph.add_edge("build_evidence", END)
    return graph.compile()


def _append_error(state: EvidenceConstructionState, message: str) -> EvidenceConstructionState:
    errors = [*state.get("errors", []), message]
    return {**state, "errors": errors}

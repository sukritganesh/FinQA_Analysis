from __future__ import annotations

import json
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.data.schemas import EvidenceUnit, FinQAExample
from src.graph.data_loading import load_examples_node, select_example_node
from src.graph.evidence import build_evidence_node
from src.graph.retrieval import build_retrieval_graph, retrieve_evidence_node
from src.retrieval.base import RetrievedEvidence, RetrievalConfig, RetrievalResult


class LoadEvidenceRetrievalState(TypedDict):
    data_path: str
    example_index: NotRequired[int]
    examples: NotRequired[list[FinQAExample]]
    selected_example: NotRequired[FinQAExample]
    question: NotRequired[str]
    evidence_units: NotRequired[list[EvidenceUnit]]
    retrieval_config: NotRequired[RetrievalConfig]
    retrieval_result: NotRequired[RetrievalResult]
    ranked_evidence: NotRequired[list[RetrievedEvidence]]
    retrieved_evidence: NotRequired[list[RetrievedEvidence]]
    errors: NotRequired[list[str]]


def test_retrieval_graph_scores_and_selects_evidence() -> None:
    graph = build_retrieval_graph()
    result = graph.invoke(
        {
            "question": "what was operating income in 2021",
            "evidence_units": [
                EvidenceUnit("text_0", "text", "company discusses operating income"),
                EvidenceUnit("table_1", "table", "revenue in 2021 is 100"),
                EvidenceUnit("table_2", "table", "operating income in 2021 is 40"),
            ],
            "retrieval_config": RetrievalConfig(mode="combined", top_k=2),
        }
    )

    assert result["errors"] == []
    assert len(result["ranked_evidence"]) == 3
    assert len(result["retrieved_evidence"]) == 2


def test_composed_graph_loads_builds_evidence_and_retrieves(tmp_path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "example-1",
                    "pre_text": ["Operating income increased."],
                    "post_text": [],
                    "table": [["Metric", "2021"], ["Operating income", "40"], ["Revenue", "100"]],
                    "qa": {"question": "what was operating income in 2021?", "exe_ans": "40"},
                }
            ]
        ),
        encoding="utf-8",
    )

    graph = _build_load_evidence_retrieval_graph()
    result = graph.invoke(
        {
            "data_path": str(path),
            "example_index": 0,
            "retrieval_config": RetrievalConfig(mode="by_source", top_k_text=1, top_k_table=1),
        }
    )
    selected_ids = {item.unit.evidence_id for item in result["retrieved_evidence"]}

    assert result["errors"] == []
    assert "text_0" in selected_ids
    assert "table_1" in selected_ids


def test_retrieval_graph_records_missing_question_error() -> None:
    graph = build_retrieval_graph()
    result = graph.invoke({"errors": []})

    assert "Missing question" in result["errors"][0]


def _add_question_from_selected_example(state: LoadEvidenceRetrievalState) -> LoadEvidenceRetrievalState:
    example = state["selected_example"]
    return {**state, "question": example.runtime.question}


def _build_load_evidence_retrieval_graph():
    graph = StateGraph(LoadEvidenceRetrievalState)
    graph.add_node("load_examples", load_examples_node)
    graph.add_node("select_example", select_example_node)
    graph.add_node("add_question", _add_question_from_selected_example)
    graph.add_node("build_evidence", build_evidence_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_edge(START, "load_examples")
    graph.add_edge("load_examples", "select_example")
    graph.add_edge("select_example", "add_question")
    graph.add_edge("add_question", "build_evidence")
    graph.add_edge("build_evidence", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", END)
    return graph.compile()

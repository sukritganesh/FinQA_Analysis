from __future__ import annotations

import json
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.data.schemas import EvidenceUnit, FinQAExample
from src.graph.data_loading import DataLoadingState, load_examples_node, select_example_node
from src.graph.evidence import (
    EvidenceConstructionState,
    build_evidence_construction_graph,
    build_evidence_node,
)


class LoadSelectEvidenceState(TypedDict):
    data_path: str
    example_index: NotRequired[int]
    example_id: NotRequired[str]
    examples: NotRequired[list[FinQAExample]]
    selected_example: NotRequired[FinQAExample]
    evidence_units: NotRequired[list[EvidenceUnit]]
    errors: NotRequired[list[str]]


def test_evidence_construction_graph_builds_evidence_for_selected_example(tmp_path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "example-1",
                    "filename": "ABC/2020/page_1.pdf",
                    "pre_text": ["Revenue increased."],
                    "post_text": ["Margin improved."],
                    "table": [["Metric", "2020"], ["Revenue", "10"]],
                    "qa": {"question": "What was revenue?", "exe_ans": "10"},
                }
            ]
        ),
        encoding="utf-8",
    )

    data_graph = _build_load_select_evidence_graph()
    result = data_graph.invoke({"data_path": str(path), "example_index": 0})
    units = {unit.evidence_id: unit for unit in result["evidence_units"]}

    assert result["errors"] == []
    assert units["text_0"].text == "Revenue increased."
    assert units["text_1"].metadata["source_section"] == "post_text"
    assert units["table_1"].text == "Metric the Revenue of 2020 is 10 ;"


def test_evidence_construction_graph_records_missing_example_error() -> None:
    graph = build_evidence_construction_graph()
    result = graph.invoke({"errors": []})

    assert "Missing selected_example" in result["errors"][0]


def _build_load_select_evidence_graph():
    graph = StateGraph(LoadSelectEvidenceState)
    graph.add_node("load_examples", load_examples_node)
    graph.add_node("select_example", select_example_node)
    graph.add_node("build_evidence", build_evidence_node)
    graph.add_edge(START, "load_examples")
    graph.add_edge("load_examples", "select_example")
    graph.add_edge("select_example", "build_evidence")
    graph.add_edge("build_evidence", END)
    return graph.compile()

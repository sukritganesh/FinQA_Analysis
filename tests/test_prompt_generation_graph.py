from __future__ import annotations

import json
from pathlib import Path
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.data.schemas import EvidenceUnit, FinQAExample
from src.graph.data_loading import load_examples_node, select_example_node
from src.graph.evidence import build_evidence_node
from src.graph.prompting import build_prompt_generation_graph, build_prompt_node
from src.graph.retrieval import retrieve_evidence_node
from src.retrieval.base import RetrievedEvidence, RetrievalConfig, RetrievalResult


class LoadEvidenceRetrievalPromptState(TypedDict):
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
    prompt_dir: NotRequired[str | Path]
    prompt: NotRequired[str]
    errors: NotRequired[list[str]]


def test_prompt_generation_graph_builds_prompt_from_question_and_retrieved_evidence(
    tmp_path: Path,
) -> None:
    prompt_dir = _write_minimal_prompt_dir(tmp_path)
    graph = build_prompt_generation_graph()
    evidence = EvidenceUnit(
        evidence_id="table_0",
        source="table",
        text="year the 2024 revenue of amount is 100 ;",
    )

    result = graph.invoke(
        {
            "question": "What was revenue in 2024?",
            "retrieved_evidence": [
                RetrievedEvidence(unit=evidence, score=3.5, rank=1, selected=True)
            ],
            "prompt_dir": prompt_dir,
        }
    )

    assert result["errors"] == []
    assert "Question: What was revenue in 2024?" in result["prompt"]
    assert "[table_0] year the 2024 revenue of amount is 100 ;" in result["prompt"]


def test_prompt_generation_graph_can_read_question_from_selected_example(tmp_path: Path) -> None:
    prompt_dir = _write_minimal_prompt_dir(tmp_path)
    graph = _build_load_evidence_retrieval_prompt_graph()
    data_path = tmp_path / "sample.json"
    data_path.write_text(
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

    result = graph.invoke(
        {
            "data_path": str(data_path),
            "example_index": 0,
            "retrieval_config": RetrievalConfig(mode="by_source", top_k_text=1, top_k_table=1),
            "prompt_dir": prompt_dir,
        }
    )

    assert result["errors"] == []
    assert "Question: what was operating income in 2021?" in result["prompt"]
    assert "[text_0] Operating income increased." in result["prompt"]
    assert "[table_1] Metric the Operating income of 2021 is 40 ;" in result["prompt"]


def test_prompt_generation_graph_records_missing_question_error() -> None:
    graph = build_prompt_generation_graph()
    result = graph.invoke({"retrieved_evidence": [], "errors": []})

    assert "Missing question" in result["errors"][0]


def test_prompt_generation_graph_records_missing_retrieved_evidence_error() -> None:
    graph = build_prompt_generation_graph()
    result = graph.invoke({"question": "What is revenue?", "errors": []})

    assert "Missing retrieved_evidence" in result["errors"][0]


def _write_minimal_prompt_dir(tmp_path: Path) -> Path:
    prompt_dir = tmp_path / "prompt"
    prompt_dir.mkdir()
    (prompt_dir / "system.txt").write_text("Return exactly one line.", encoding="utf-8")
    (prompt_dir / "task_template.txt").write_text(
        "Question: {question}\nEvidence:\n{evidence_context}",
        encoding="utf-8",
    )
    return prompt_dir


def _add_question_from_selected_example(
    state: LoadEvidenceRetrievalPromptState,
) -> LoadEvidenceRetrievalPromptState:
    example = state["selected_example"]
    return {**state, "question": example.runtime.question}


def _build_load_evidence_retrieval_prompt_graph():
    graph = StateGraph(LoadEvidenceRetrievalPromptState)
    graph.add_node("load_examples", load_examples_node)
    graph.add_node("select_example", select_example_node)
    graph.add_node("add_question", _add_question_from_selected_example)
    graph.add_node("build_evidence", build_evidence_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("build_prompt", build_prompt_node)
    graph.add_edge(START, "load_examples")
    graph.add_edge("load_examples", "select_example")
    graph.add_edge("select_example", "add_question")
    graph.add_edge("add_question", "build_evidence")
    graph.add_edge("build_evidence", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "build_prompt")
    graph.add_edge("build_prompt", END)
    return graph.compile()

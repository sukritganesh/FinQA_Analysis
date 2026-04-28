from __future__ import annotations

import json

from src.graph.data_loading import build_data_loading_graph


def test_data_loading_graph_loads_and_selects_by_index(tmp_path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "example-1",
                    "filename": "ABC/2020/page_1.pdf",
                    "pre_text": ["Revenue increased."],
                    "post_text": [],
                    "table": [["Metric", "2020"], ["Revenue", "10"]],
                    "qa": {"question": "What was revenue?", "exe_ans": "10"},
                },
                {
                    "id": "example-2",
                    "pre_text": [],
                    "post_text": [],
                    "table": [],
                    "qa": {"question": "What was margin?"},
                },
            ]
        ),
        encoding="utf-8",
    )

    graph = build_data_loading_graph()
    result = graph.invoke({"data_path": str(path), "example_index": 1})

    assert result["errors"] == []
    assert len(result["examples"]) == 2
    assert result["selected_example"].runtime.example_id == "example-2"


def test_data_loading_graph_loads_and_selects_by_id(tmp_path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "example-1",
                    "pre_text": [],
                    "post_text": [],
                    "table": [],
                    "qa": {"question": "Q1"},
                },
                {
                    "id": "example-2",
                    "pre_text": [],
                    "post_text": [],
                    "table": [],
                    "qa": {"question": "Q2"},
                },
            ]
        ),
        encoding="utf-8",
    )

    graph = build_data_loading_graph()
    result = graph.invoke({"data_path": str(path), "example_id": "example-2"})

    assert result["selected_example"].runtime.question == "Q2"


def test_data_loading_graph_records_selection_errors(tmp_path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "example-1",
                    "pre_text": [],
                    "post_text": [],
                    "table": [],
                    "qa": {"question": "Q1"},
                }
            ]
        ),
        encoding="utf-8",
    )

    graph = build_data_loading_graph()
    result = graph.invoke({"data_path": str(path), "example_index": 4})

    assert "out of range" in result["errors"][0]

from __future__ import annotations

import json

from src.data.loader import load_finqa_examples, normalize_finqa_example


def test_normalize_finqa_example_separates_runtime_and_gold() -> None:
    raw = {
        "id": "example-1",
        "filename": "ABC/2020/page_1.pdf",
        "pre_text": ["Revenue was up."],
        "post_text": ["Net income was stable."],
        "table_ori": [["Metric", "2020"], ["Revenue", "$10"]],
        "table": [["Metric", "2020"], ["Revenue", "10"]],
        "text_retrieved": [{"score": 1.0, "ind": "text_0"}],
        "qa": {
            "question": "What was revenue?",
            "answer": "10",
            "program": "table_max(10)",
            "gold_inds": {"text_0": "Revenue was up."},
            "exe_ans": "10",
            "program_re": "table_max(10)",
            "ann_text_rows": [0],
            "ann_table_rows": [],
            "steps": [{"op": "table_max", "arg1": "10", "res": "10"}],
        },
    }

    example = normalize_finqa_example(raw)

    assert example.runtime.example_id == "example-1"
    assert example.runtime.filename == "ABC/2020/page_1.pdf"
    assert example.runtime.question == "What was revenue?"
    assert example.gold.answer == "10"
    assert example.gold.executable_answer == "10"
    assert example.gold.program == "table_max(10)"
    assert example.gold.supporting_facts == {"text_0": "Revenue was up."}
    assert example.gold.ann_text_rows == [0]
    assert example.gold.has_labels is True
    assert example.metadata.table_original == [["Metric", "2020"], ["Revenue", "$10"]]
    assert example.metadata.text_retrieved == [{"score": 1.0, "ind": "text_0"}]


def test_load_finqa_examples_reads_json_list(tmp_path) -> None:
    payload = [
        {
            "id": "example-1",
            "pre_text": [],
            "post_text": [],
            "table": [],
            "qa": {"question": "Q1", "exe_ans": "1"},
        }
    ]
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    examples = load_finqa_examples(path)

    assert len(examples) == 1
    assert examples[0].runtime.question == "Q1"


def test_normalize_finqa_example_handles_private_test_style_record() -> None:
    raw = {
        "id": "private-1",
        "filename": "ETR/2011/page_301.pdf",
        "pre_text": ["Management discussion."],
        "post_text": ["See note 4."],
        "table": [["2011", "2010"], ["23596", "63003"]],
        "table_ori": [["2011", "2010"], ["$ 23596", "$ 63003"]],
        "qa": {
            "question": "What is the percentage change?",
        },
    }

    example = normalize_finqa_example(raw)

    assert example.runtime.example_id == "private-1"
    assert example.runtime.question == "What is the percentage change?"
    assert example.gold.has_labels is False
    assert example.gold.executable_answer is None
    assert example.gold.supporting_facts == {}
    assert example.metadata.table_original == [["2011", "2010"], ["$ 23596", "$ 63003"]]


def test_load_finqa_examples_rejects_non_list_top_level(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"id": "not-a-list"}), encoding="utf-8")

    try:
        load_finqa_examples(path)
    except ValueError as exc:
        assert "Expected top-level list" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-list top-level JSON.")

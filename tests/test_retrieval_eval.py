from __future__ import annotations

from src.data.loader import load_finqa_examples
from src.eval.retrieval import evaluate_retrieval_on_examples, run_retrieval_evaluation
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.base import RetrievalConfig


def test_evaluate_retrieval_on_examples_compares_selected_ids_to_gold_support() -> None:
    examples = [
        example
        for example in load_finqa_examples("data/raw/test.json")
        if example.runtime.example_id
        in {
            "ETR/2016/page_23.pdf-2",
            "INTC/2015/page_41.pdf-4",
            "FIS/2010/page_70.pdf-2",
        }
    ]

    summary = evaluate_retrieval_on_examples(
        examples=examples,
        retriever=BM25Retriever(),
        config=RetrievalConfig(mode="by_source", top_k_text=3, top_k_table=3),
    )

    assert summary.total_examples == 3
    assert summary.examples_with_gold == 3
    assert summary.examples_with_hit == 3
    assert summary.recall_at_selection == 1.0


def test_run_retrieval_evaluation_supports_limit_config_and_log_output(tmp_path) -> None:
    log_path = tmp_path / "retrieval_eval.md"

    report = run_retrieval_evaluation(
        data_path="data/raw/test.json",
        config=RetrievalConfig(mode="by_source", top_k_text=3, top_k_table=3),
        limit=25,
        log_path=log_path,
    )
    log_text = log_path.read_text(encoding="utf-8")

    assert report.summary.total_examples == 25
    assert report.summary.examples_with_gold == 25
    assert report.summary.examples_with_hit >= 20
    assert report.summary.recall_at_selection >= 0.8
    assert len(report.details) == 25
    assert log_path.exists()
    assert "# Retrieval Evaluation Report" in log_text
    assert "recall_at_selection" in log_text
    assert "Selected evidence preview" in log_text
    assert "Missing gold evidence preview" in log_text
    assert "rank=" in log_text


def test_run_retrieval_evaluation_can_include_hit_details_in_log(tmp_path) -> None:
    log_path = tmp_path / "retrieval_eval_with_hits.md"

    run_retrieval_evaluation(
        data_path="data/raw/test.json",
        config=RetrievalConfig(mode="by_source", top_k_text=3, top_k_table=3),
        limit=3,
        log_path=log_path,
        include_hits_in_log=True,
    )
    log_text = log_path.read_text(encoding="utf-8")

    assert "## Hits" in log_text
    assert "### HIT:" in log_text
    assert "GOLD_MATCH" in log_text

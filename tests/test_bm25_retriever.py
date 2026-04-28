from __future__ import annotations

from src.data.evidence import build_evidence_units
from src.data.loader import load_finqa_examples
from src.data.schemas import EvidenceUnit
from src.retrieval.bm25 import BM25Retriever, tokenize_for_bm25
from src.retrieval.base import RetrievalConfig


def test_tokenize_for_bm25_normalizes_text() -> None:
    assert tokenize_for_bm25("Revenue, 2021!") == ["revenue", "2021"]


def test_bm25_combined_mode_scores_all_and_selects_top_k() -> None:
    evidence_units = [
        EvidenceUnit("text_0", "text", "cash flow commentary"),
        EvidenceUnit("table_1", "table", "Metric the Revenue of 2021 is 100 ;"),
        EvidenceUnit("table_2", "table", "Metric the Operating income of 2021 is 40 ;"),
    ]

    result = BM25Retriever().retrieve(
        question="what was operating income in 2021",
        evidence_units=evidence_units,
        config=RetrievalConfig(mode="combined", top_k=1),
    )

    assert len(result.ranked_evidence) == 3
    assert len(result.selected_evidence) == 1
    assert result.selected_evidence[0].unit.evidence_id == "table_2"
    assert result.selected_evidence[0].selected is True


def test_bm25_by_source_mode_selects_text_and_table_quotas() -> None:
    evidence_units = [
        EvidenceUnit("text_0", "text", "company discusses operating income"),
        EvidenceUnit("text_1", "text", "unrelated cash flow sentence"),
        EvidenceUnit("table_1", "table", "Metric the Revenue of 2021 is 100 ;"),
        EvidenceUnit("table_2", "table", "Metric the Operating income of 2021 is 40 ;"),
    ]

    result = BM25Retriever().retrieve(
        question="what was operating income in 2021",
        evidence_units=evidence_units,
        config=RetrievalConfig(mode="by_source", top_k_text=1, top_k_table=1),
    )

    selected_ids = [item.unit.evidence_id for item in result.selected_evidence]

    assert len(result.ranked_evidence) == 4
    assert selected_ids == ["text_0", "table_2"]
    assert all(item.selected for item in result.selected_evidence)


def test_bm25_retriever_scores_real_finqa_evidence_from_test_json() -> None:
    examples = {
        example.runtime.example_id: example
        for example in load_finqa_examples("data/raw/test.json")
    }
    example = examples["ETR/2016/page_23.pdf-2"]
    evidence_units = build_evidence_units(example)

    result = BM25Retriever().retrieve(
        question=example.runtime.question,
        evidence_units=evidence_units,
        config=RetrievalConfig(mode="by_source", top_k_text=3, top_k_table=3),
    )

    selected_ids = {item.unit.evidence_id for item in result.selected_evidence}

    assert len(result.ranked_evidence) == len(evidence_units)
    assert {"table_1", "table_8"}.issubset(selected_ids)


def test_bm25_modes_retrieve_gold_support_for_representative_real_examples() -> None:
    examples = {
        example.runtime.example_id: example
        for example in load_finqa_examples("data/raw/test.json")
    }
    example_ids = [
        "ETR/2016/page_23.pdf-2",
        "INTC/2015/page_41.pdf-4",
        "FIS/2010/page_70.pdf-2",
    ]

    for example_id in example_ids:
        example = examples[example_id]
        gold_ids = set(example.gold.supporting_facts)
        evidence_units = build_evidence_units(example)

        by_source_result = BM25Retriever().retrieve(
            question=example.runtime.question,
            evidence_units=evidence_units,
            config=RetrievalConfig(mode="by_source", top_k_text=3, top_k_table=3),
        )
        combined_result = BM25Retriever().retrieve(
            question=example.runtime.question,
            evidence_units=evidence_units,
            config=RetrievalConfig(mode="combined", top_k=6),
        )

        by_source_ids = {item.unit.evidence_id for item in by_source_result.selected_evidence}
        combined_ids = {item.unit.evidence_id for item in combined_result.selected_evidence}

        assert gold_ids & by_source_ids, f"by_source missed gold support for {example_id}"
        assert gold_ids & combined_ids, f"combined missed gold support for {example_id}"

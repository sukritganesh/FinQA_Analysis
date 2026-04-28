"""BM25 retrieval over FinQA evidence units."""

from __future__ import annotations

from collections.abc import Iterable

from rank_bm25 import BM25Okapi

from src.data.schemas import EvidenceUnit
from src.retrieval.base import RetrievedEvidence, RetrievalConfig, RetrievalResult, Retriever
from src.utils.text import normalize_for_matching


class BM25Retriever(Retriever):
    """BM25 retriever with combined and by-source selection modes."""

    def retrieve(
        self,
        question: str,
        evidence_units: list[EvidenceUnit],
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        """Score all evidence and select top-k according to the config."""
        config = config or RetrievalConfig()

        if config.strategy != "bm25":
            msg = f"BM25Retriever only supports strategy='bm25', got {config.strategy!r}"
            raise ValueError(msg)
        if config.mode == "combined":
            return self._retrieve_combined(question, evidence_units, config)
        if config.mode == "by_source":
            return self._retrieve_by_source(question, evidence_units, config)

        msg = f"Unsupported retrieval mode: {config.mode}"
        raise ValueError(msg)

    def _retrieve_combined(
        self,
        question: str,
        evidence_units: list[EvidenceUnit],
        config: RetrievalConfig,
    ) -> RetrievalResult:
        ranked = _score_pool(question, evidence_units)
        selected_ids = {item.unit.evidence_id for item in ranked[: config.top_k]}
        selected = _mark_selected(ranked, selected_ids)
        return RetrievalResult(
            ranked_evidence=selected,
            selected_evidence=[item for item in selected if item.selected],
            config=config,
        )

    def _retrieve_by_source(
        self,
        question: str,
        evidence_units: list[EvidenceUnit],
        config: RetrievalConfig,
    ) -> RetrievalResult:
        text_ranked = _score_pool(question, [unit for unit in evidence_units if unit.source == "text"])
        table_ranked = _score_pool(question, [unit for unit in evidence_units if unit.source == "table"])

        selected_ids = {
            item.unit.evidence_id
            for item in [*text_ranked[: config.top_k_text], *table_ranked[: config.top_k_table]]
        }
        ranked = _merge_source_rankings(text_ranked, table_ranked)
        ranked = _mark_selected(ranked, selected_ids)
        selected = [item for item in [*text_ranked[: config.top_k_text], *table_ranked[: config.top_k_table]]]
        selected = _mark_selected(selected, selected_ids)

        return RetrievalResult(
            ranked_evidence=ranked,
            selected_evidence=selected,
            config=config,
        )


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for lightweight BM25 retrieval."""
    return normalize_for_matching(text).split()


def _score_pool(question: str, evidence_units: list[EvidenceUnit]) -> list[RetrievedEvidence]:
    if not evidence_units:
        return []

    tokenized_corpus = [tokenize_for_bm25(unit.text) for unit in evidence_units]
    query_tokens = tokenize_for_bm25(question)

    if not query_tokens or not any(tokenized_corpus):
        scores = [0.0 for _ in evidence_units]
    else:
        bm25 = BM25Okapi(tokenized_corpus)
        scores = [float(score) for score in bm25.get_scores(query_tokens)]

    ranked = [
        RetrievedEvidence(
            unit=unit,
            score=score,
            rank=0,
            source_rank=None,
            selected=False,
            metadata={"retrieval_strategy": "bm25"},
        )
        for unit, score in zip(evidence_units, scores, strict=True)
    ]
    ranked.sort(key=lambda item: (-item.score, item.unit.evidence_id))

    for rank, item in enumerate(ranked, start=1):
        item.rank = rank
        item.source_rank = rank

    return ranked


def _merge_source_rankings(*rankings: Iterable[RetrievedEvidence]) -> list[RetrievedEvidence]:
    merged = [item for ranking in rankings for item in ranking]
    merged.sort(key=lambda item: (-item.score, item.unit.source, item.unit.evidence_id))
    for rank, item in enumerate(merged, start=1):
        item.rank = rank
    return merged


def _mark_selected(items: list[RetrievedEvidence], selected_ids: set[str]) -> list[RetrievedEvidence]:
    for item in items:
        item.selected = item.unit.evidence_id in selected_ids
    return items

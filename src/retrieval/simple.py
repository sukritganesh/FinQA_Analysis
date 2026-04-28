"""Simple baseline retriever implementations."""

from __future__ import annotations

from collections import Counter

from src.data.schemas import EvidenceUnit
from src.retrieval.base import RetrievedEvidence, RetrievalConfig, RetrievalResult, Retriever
from src.utils.text import normalize_for_matching


class KeywordOverlapRetriever(Retriever):
    """A tiny, inspectable lexical baseline for early development."""

    def retrieve(
        self,
        question: str,
        evidence_units: list[EvidenceUnit],
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        config = config or RetrievalConfig(strategy="bm25")
        question_tokens = _token_counts(question)
        scored: list[RetrievedEvidence] = []

        for unit in evidence_units:
            evidence_tokens = _token_counts(unit.text)
            score = _overlap_score(question_tokens, evidence_tokens)
            scored.append(RetrievedEvidence(unit=unit, score=score, rank=0))

        scored.sort(key=lambda item: (-item.score, item.unit.evidence_id))
        for rank, item in enumerate(scored, start=1):
            item.rank = rank
            item.selected = rank <= config.top_k

        return RetrievalResult(
            ranked_evidence=scored,
            selected_evidence=[item for item in scored if item.selected],
            config=config,
        )


def _token_counts(text: str) -> Counter[str]:
    normalized = normalize_for_matching(text)
    return Counter(token for token in normalized.split() if token)


def _overlap_score(left: Counter[str], right: Counter[str]) -> float:
    overlap = sum((left & right).values())
    return float(overlap)

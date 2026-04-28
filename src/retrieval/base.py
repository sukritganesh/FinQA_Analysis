"""Retriever interfaces for evidence selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from src.data.schemas import EvidenceUnit

RetrievalMode = Literal["combined", "by_source"]
RetrievalStrategy = Literal["bm25"]


@dataclass(slots=True)
class RetrievalConfig:
    """Configuration shared by retrieval implementations."""

    strategy: RetrievalStrategy = "bm25"
    mode: RetrievalMode = "by_source"
    top_k: int = 5
    top_k_text: int = 3
    top_k_table: int = 3


@dataclass(slots=True)
class RetrievedEvidence:
    """A scored evidence item returned by a retriever."""

    unit: EvidenceUnit
    score: float
    rank: int
    source_rank: int | None = None
    selected: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """Full retrieval output with all scored evidence and selected top-k evidence."""

    ranked_evidence: list[RetrievedEvidence]
    selected_evidence: list[RetrievedEvidence]
    config: RetrievalConfig


class Retriever(Protocol):
    """Protocol for swappable retrieval implementations."""

    def retrieve(
        self,
        question: str,
        evidence_units: list[EvidenceUnit],
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        """Score evidence units and return retrieval results."""

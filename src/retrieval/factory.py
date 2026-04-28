"""Factory helpers for choosing retrieval implementations."""

from __future__ import annotations

from src.retrieval.base import Retriever, RetrievalStrategy
from src.retrieval.bm25 import BM25Retriever


def build_retriever(strategy: RetrievalStrategy) -> Retriever:
    """Build a retriever for a configured strategy."""
    if strategy == "bm25":
        return BM25Retriever()

    msg = f"Unsupported retrieval strategy: {strategy}"
    raise ValueError(msg)

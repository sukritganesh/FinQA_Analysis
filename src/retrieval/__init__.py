"""Retrieval interfaces and baseline implementations."""

from src.retrieval.base import RetrievedEvidence, RetrievalConfig, RetrievalResult, Retriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.factory import build_retriever

__all__ = [
    "BM25Retriever",
    "RetrievedEvidence",
    "RetrievalConfig",
    "RetrievalResult",
    "Retriever",
    "build_retriever",
]

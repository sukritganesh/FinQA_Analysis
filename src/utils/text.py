"""Basic text normalization helpers."""

from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into rough sentence-like units."""
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]


def normalize_for_matching(text: str) -> str:
    """Lowercase and strip punctuation for lightweight lexical matching."""
    normalized = normalize_whitespace(text).lower()
    normalized = _NON_ALNUM_RE.sub(" ", normalized)
    return normalize_whitespace(normalized)

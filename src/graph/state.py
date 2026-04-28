"""Shared state objects for the FinQA workflow."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data.schemas import EvidenceUnit, FinQAExample
from src.llm.parser import ParsedReasoningOutput
from src.retrieval.base import RetrievedEvidence


@dataclass(slots=True)
class PipelineState:
    """Mutable state passed between workflow stages."""

    example: FinQAExample
    evidence_units: list[EvidenceUnit] = field(default_factory=list)
    retrieved_evidence: list[RetrievedEvidence] = field(default_factory=list)
    prompt: str | None = None
    model_output_text: str | None = None
    parsed_output: ParsedReasoningOutput | None = None
    final_answer: str | None = None
    errors: list[str] = field(default_factory=list)

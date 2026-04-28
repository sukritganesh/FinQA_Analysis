"""Prediction objects produced by the FinQA inference pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.data.schemas import FinQAExample
from src.eval.metrics import normalize_answer_text
from src.llm.parser import ParsedReasoningOutput
from src.retrieval.base import RetrievedEvidence
from src.tools.executor import ExecutionResult


@dataclass(frozen=True, slots=True)
class Prediction:
    """Evaluation-ready result for one FinQA example."""

    example_id: str | None
    question: str | None
    answer: str | None
    normalized_answer: str | None
    model_output_text: str | None
    parsed_output: ParsedReasoningOutput | None = None
    execution_result: ExecutionResult | None = None
    selected_evidence: dict[str, str] = field(default_factory=dict)
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_answer(self) -> bool:
        """Return whether the pipeline produced a final answer."""
        return self.answer is not None

    @property
    def is_successful(self) -> bool:
        """Return whether the prediction has an answer and no recorded errors."""
        return self.has_answer and not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prediction to plain Python containers."""
        return asdict(self)


def build_prediction(
    *,
    selected_example: FinQAExample | None,
    final_answer: str | None,
    model_output_text: str | None,
    parsed_output: ParsedReasoningOutput | None,
    execution_result: ExecutionResult | None,
    selected_evidence: list[RetrievedEvidence] | tuple[RetrievedEvidence, ...] | None = None,
    errors: list[str] | tuple[str, ...] | None = None,
) -> Prediction:
    """Build a normalized prediction object from pipeline state artifacts."""
    return Prediction(
        example_id=_get_example_id(selected_example),
        question=_get_question(selected_example),
        answer=final_answer,
        normalized_answer=normalize_answer_text(final_answer),
        model_output_text=model_output_text,
        parsed_output=parsed_output,
        execution_result=execution_result,
        selected_evidence=_selected_evidence_record(selected_evidence or ()),
        errors=tuple(errors or ()),
    )


def _get_example_id(example: FinQAExample | None) -> str | None:
    if example is None:
        return None
    return example.runtime.example_id


def _get_question(example: FinQAExample | None) -> str | None:
    if example is None:
        return None
    return example.runtime.question


def _selected_evidence_record(
    selected_evidence: list[RetrievedEvidence] | tuple[RetrievedEvidence, ...],
) -> dict[str, str]:
    return {item.unit.evidence_id: item.unit.text for item in selected_evidence}

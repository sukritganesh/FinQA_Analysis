"""Answer normalization and basic evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from src.tools.calculator import parse_decimal
from src.utils.text import normalize_for_matching


DEFAULT_ABSOLUTE_TOLERANCE = Decimal("0.0001")
DEFAULT_RELATIVE_TOLERANCE = Decimal("0.0001")


@dataclass(slots=True)
class AnswerMatchResult:
    """Simple answer comparison result."""

    prediction: str | None
    gold: str | None
    is_correct: bool
    normalized_prediction: str | None = None
    normalized_gold: str | None = None
    match_type: str = "exact"
    note: str | None = None


def normalize_answer_text(value: str | None) -> str | None:
    """Normalize an answer string for rough matching."""
    if value is None:
        return None
    return normalize_for_matching(value)


def exact_match(prediction: str | None, gold: str | None) -> AnswerMatchResult:
    """Compute a normalized exact match score."""
    normalized_prediction = normalize_answer_text(prediction)
    normalized_gold = normalize_answer_text(gold)
    return AnswerMatchResult(
        prediction=prediction,
        gold=gold,
        is_correct=normalized_prediction == normalized_gold,
        normalized_prediction=normalized_prediction,
        normalized_gold=normalized_gold,
    )


def finqa_answer_match(
    prediction: str | None,
    gold: str | None,
    *,
    absolute_tolerance: Decimal = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: Decimal = DEFAULT_RELATIVE_TOLERANCE,
    allow_percent_prediction: bool = True,
) -> AnswerMatchResult:
    """Compare a FinQA prediction to a gold execution answer."""
    normalized_prediction = normalize_answer_text(prediction)
    normalized_gold = normalize_answer_text(gold)

    if normalized_gold is None:
        return AnswerMatchResult(
            prediction=prediction,
            gold=gold,
            is_correct=False,
            normalized_prediction=normalized_prediction,
            normalized_gold=normalized_gold,
            match_type="missing_gold",
            note="Gold answer is missing.",
        )

    if normalized_prediction is None:
        return AnswerMatchResult(
            prediction=prediction,
            gold=gold,
            is_correct=False,
            normalized_prediction=normalized_prediction,
            normalized_gold=normalized_gold,
            match_type="missing_prediction",
            note="Prediction answer is missing.",
        )

    if normalized_gold in {"yes", "no"} or normalized_prediction in {"yes", "no"}:
        return AnswerMatchResult(
            prediction=prediction,
            gold=gold,
            is_correct=normalized_prediction == normalized_gold,
            normalized_prediction=normalized_prediction,
            normalized_gold=normalized_gold,
            match_type="yes_no",
        )

    predicted_number, prediction_note = _parse_eval_decimal(
        prediction,
        allow_percent=allow_percent_prediction,
    )
    gold_number, _ = _parse_eval_decimal(gold, allow_percent=False)

    if predicted_number is None or gold_number is None:
        return AnswerMatchResult(
            prediction=prediction,
            gold=gold,
            is_correct=normalized_prediction == normalized_gold,
            normalized_prediction=normalized_prediction,
            normalized_gold=normalized_gold,
            match_type="exact_fallback",
            note="Could not parse both answers as numbers.",
        )

    difference = abs(predicted_number - gold_number)
    tolerance = max(absolute_tolerance, relative_tolerance * max(Decimal("1"), abs(gold_number)))
    return AnswerMatchResult(
        prediction=prediction,
        gold=gold,
        is_correct=difference <= tolerance,
        normalized_prediction=format(predicted_number.normalize(), "f"),
        normalized_gold=format(gold_number.normalize(), "f"),
        match_type="numeric_tolerance",
        note=prediction_note,
    )


def _parse_eval_decimal(value: str | None, *, allow_percent: bool) -> tuple[Decimal | None, str | None]:
    if value is None:
        return None, None

    note = None
    text = value.strip()
    if text.endswith("%"):
        if not allow_percent:
            return None, None
        text = text[:-1].strip()
        note = "Converted percent prediction to decimal."
        divisor = Decimal("100")
    else:
        divisor = Decimal("1")

    try:
        return parse_decimal(text) / divisor, note
    except (InvalidOperation, ValueError):
        return None, None

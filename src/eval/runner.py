"""Evaluation entrypoints for comparing predictions to gold examples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.data.loader import load_finqa_examples
from src.eval.metrics import exact_match
from src.utils.io import read_json


@dataclass(slots=True)
class EvaluationSummary:
    """High-level answer evaluation summary."""

    total_examples: int
    matched_examples: int
    accuracy: float
    missing_predictions: list[str]

    def to_pretty_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_prediction_map(path: str | Path) -> dict[str, str]:
    """Load a minimal id-to-answer prediction mapping from JSON."""
    payload = read_json(path)
    if not isinstance(payload, dict):
        msg = "Predictions file must contain an object mapping example id to answer."
        raise ValueError(msg)
    return {str(key): str(value) for key, value in payload.items()}


def run_answer_evaluation(gold_path: str | Path, predictions: dict[str, str]) -> EvaluationSummary:
    """Compare predicted answers to normalized gold execution answers."""
    examples = load_finqa_examples(gold_path)
    matched = 0
    missing_predictions: list[str] = []

    for example in examples:
        example_id = example.runtime.example_id
        prediction = predictions.get(example_id)
        if prediction is None:
            missing_predictions.append(example_id)
            continue

        gold_answer = example.gold.executable_answer or example.gold.answer
        result = exact_match(prediction=prediction, gold=gold_answer)
        if result.is_correct:
            matched += 1

    total = len(examples)
    accuracy = matched / total if total else 0.0
    return EvaluationSummary(
        total_examples=total,
        matched_examples=matched,
        accuracy=accuracy,
        missing_predictions=missing_predictions,
    )

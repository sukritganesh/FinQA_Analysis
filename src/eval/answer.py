"""Final-answer evaluation for FinQA predictions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.data.schemas import FinQAExample
from src.eval.metrics import AnswerMatchResult, finqa_answer_match
from src.eval.prediction import Prediction


@dataclass(frozen=True, slots=True)
class AnswerEvaluationResult:
    """Single-example answer evaluation result."""

    example_id: str
    question: str
    has_gold: bool
    is_correct: bool
    prediction_answer: str | None
    gold_answer: str | None
    prediction_program: str | None
    gold_program: str | None
    gold_inds: dict[str, str]
    selected_evidence: dict[str, str]
    retrieved_all_gold_inds: bool | None
    match: AnswerMatchResult | None
    prediction_errors: tuple[str, ...]

    def to_dict(self) -> dict:
        """Serialize this result to plain Python containers."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BatchEvaluationSummary:
    """Aggregate answer-evaluation counts for a prediction batch."""

    total_examples: int
    examples_with_gold: int
    examples_without_gold: int
    answered_examples: int
    unanswered_examples: int
    correct_answers: int
    accuracy: float | None
    missing_predictions: int
    parse_failures: int
    execution_failures: int
    other_pipeline_errors: int
    scored: bool

    def to_dict(self) -> dict:
        """Serialize this summary to plain Python containers."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BatchEvaluationReport:
    """Batch evaluation report with summary and per-example details."""

    summary: BatchEvaluationSummary
    details: tuple[AnswerEvaluationResult, ...]

    def to_dict(self) -> dict:
        """Serialize this report to plain Python containers."""
        return {
            "summary": self.summary.to_dict(),
            "details": [detail.to_dict() for detail in self.details],
        }


def evaluate_prediction_answer(
    prediction: Prediction,
    example: FinQAExample,
) -> AnswerEvaluationResult:
    """Evaluate one prediction against one FinQA gold answer if available."""
    gold_answer = example.gold.executable_answer or example.gold.answer
    if gold_answer is None:
        return AnswerEvaluationResult(
            example_id=example.runtime.example_id,
            question=example.runtime.question,
            has_gold=False,
            is_correct=False,
            prediction_answer=prediction.answer,
            gold_answer=None,
            prediction_program=prediction.model_output_text,
            gold_program=example.gold.program,
            gold_inds=dict(example.gold.supporting_facts),
            selected_evidence=prediction.selected_evidence,
            retrieved_all_gold_inds=_retrieved_all_gold_inds(prediction, example),
            match=None,
            prediction_errors=prediction.errors,
        )

    match = finqa_answer_match(prediction=prediction.answer, gold=gold_answer)
    return AnswerEvaluationResult(
        example_id=example.runtime.example_id,
        question=example.runtime.question,
        has_gold=True,
        is_correct=match.is_correct,
        prediction_answer=prediction.answer,
        gold_answer=gold_answer,
        prediction_program=prediction.model_output_text,
        gold_program=example.gold.program,
        gold_inds=dict(example.gold.supporting_facts),
        selected_evidence=prediction.selected_evidence,
        retrieved_all_gold_inds=_retrieved_all_gold_inds(prediction, example),
        match=match,
        prediction_errors=prediction.errors,
    )


def evaluate_prediction_batch(
    predictions: Iterable[Prediction],
    examples: Iterable[FinQAExample],
) -> BatchEvaluationReport:
    """Evaluate many predictions against matching FinQA examples."""
    prediction_by_id = {
        prediction.example_id: prediction
        for prediction in predictions
        if prediction.example_id is not None
    }

    details: list[AnswerEvaluationResult] = []
    total_examples = 0
    examples_with_gold = 0
    examples_without_gold = 0
    answered_examples = 0
    correct_answers = 0
    missing_predictions = 0
    parse_failures = 0
    execution_failures = 0
    other_pipeline_errors = 0

    for example in examples:
        total_examples += 1
        example_id = example.runtime.example_id
        has_gold = (example.gold.executable_answer or example.gold.answer) is not None
        if has_gold:
            examples_with_gold += 1
        else:
            examples_without_gold += 1

        prediction = prediction_by_id.get(example_id)
        if prediction is None:
            missing_predictions += 1
            detail = _missing_prediction_detail(example)
            details.append(detail)
            continue

        if prediction.has_answer:
            answered_examples += 1

        error_counts = _count_prediction_errors(prediction.errors)
        parse_failures += error_counts["parse"]
        execution_failures += error_counts["execution"]
        other_pipeline_errors += error_counts["other"]

        detail = evaluate_prediction_answer(prediction, example)
        if detail.has_gold and detail.is_correct:
            correct_answers += 1
        details.append(detail)

    accuracy = correct_answers / examples_with_gold if examples_with_gold else None
    summary = BatchEvaluationSummary(
        total_examples=total_examples,
        examples_with_gold=examples_with_gold,
        examples_without_gold=examples_without_gold,
        answered_examples=answered_examples,
        unanswered_examples=total_examples - answered_examples,
        correct_answers=correct_answers,
        accuracy=accuracy,
        missing_predictions=missing_predictions,
        parse_failures=parse_failures,
        execution_failures=execution_failures,
        other_pipeline_errors=other_pipeline_errors,
        scored=examples_with_gold > 0,
    )
    return BatchEvaluationReport(summary=summary, details=tuple(details))


def write_batch_evaluation_outputs(
    report: BatchEvaluationReport,
    *,
    output_dir: str | Path,
    run_name: str,
) -> dict[str, Path]:
    """Write summary, detail JSON, and a small Markdown details file."""
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = resolved_output_dir / f"{run_name}_summary.json"
    details_path = resolved_output_dir / f"{run_name}_details.json"
    markdown_path = resolved_output_dir / f"{run_name}_details.md"

    _write_json(summary_path, report.summary.to_dict())
    _write_json(details_path, [detail.to_dict() for detail in report.details])
    markdown_path.write_text(_render_markdown_details(report), encoding="utf-8")

    return {
        "summary": summary_path,
        "details": details_path,
        "markdown": markdown_path,
    }


def _missing_prediction_detail(example: FinQAExample) -> AnswerEvaluationResult:
    gold_answer = example.gold.executable_answer or example.gold.answer
    return AnswerEvaluationResult(
        example_id=example.runtime.example_id,
        question=example.runtime.question,
        has_gold=gold_answer is not None,
        is_correct=False,
        prediction_answer=None,
        gold_answer=gold_answer,
        prediction_program=None,
        gold_program=example.gold.program,
        gold_inds=dict(example.gold.supporting_facts),
        selected_evidence={},
        retrieved_all_gold_inds=None if not example.gold.supporting_facts else False,
        match=None,
        prediction_errors=("Missing prediction.",),
    )


def _retrieved_all_gold_inds(prediction: Prediction, example: FinQAExample) -> bool | None:
    gold_ids = set(example.gold.supporting_facts)
    if not gold_ids:
        return None
    selected_ids = set(prediction.selected_evidence)
    return gold_ids.issubset(selected_ids)


def _count_prediction_errors(errors: tuple[str, ...]) -> dict[str, int]:
    counts = {"parse": 0, "execution": 0, "other": 0}
    for error in errors:
        normalized = error.lower()
        if "execute" in normalized or "execution" in normalized:
            counts["execution"] += 1
        elif "parse" in normalized:
            counts["parse"] += 1
        else:
            counts["other"] += 1
    return counts


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _render_markdown_details(report: BatchEvaluationReport) -> str:
    lines = [
        "# Answer Evaluation Details",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report.summary.to_dict(), indent=2),
        "```",
        "",
        "## Examples",
        "",
        "| example_id | question | prediction | gold | predicted_program | gold_program | retrieved_all_gold_inds | correct | match_type | errors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for detail in report.details:
        match_type = detail.match.match_type if detail.match is not None else ""
        errors = "; ".join(detail.prediction_errors)
        lines.append(
            (
                "| {example_id} | {question} | {prediction} | {gold} | {predicted_program} | "
                "{gold_program} | {retrieved_all_gold_inds} | {correct} | {match_type} | {errors} |"
            ).format(
                example_id=_escape_markdown_cell(detail.example_id),
                question=_escape_markdown_cell(detail.question),
                prediction=_escape_markdown_cell(detail.prediction_answer),
                gold=_escape_markdown_cell(detail.gold_answer),
                predicted_program=_escape_markdown_cell(detail.prediction_program),
                gold_program=_escape_markdown_cell(detail.gold_program),
                retrieved_all_gold_inds=_escape_markdown_cell(detail.retrieved_all_gold_inds),
                correct="yes" if detail.is_correct else "no",
                match_type=_escape_markdown_cell(match_type),
                errors=_escape_markdown_cell(errors),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _escape_markdown_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")

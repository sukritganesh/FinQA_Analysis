from __future__ import annotations

import json

from src.data.loader import load_finqa_examples
from src.data.schemas import EvidenceUnit, FinQAExample, RuntimeInputs
from src.eval.answer import (
    evaluate_prediction_answer,
    evaluate_prediction_batch,
    write_batch_evaluation_outputs,
)
from src.eval.metrics import exact_match, finqa_answer_match
from src.eval.prediction import build_prediction
from src.retrieval.base import RetrievedEvidence

def test_finqa_answer_match_integers() -> None:
    result = finqa_answer_match(prediction="5.0", gold="5")

    assert result.is_correct
    assert result.match_type == "numeric_tolerance"

def test_finqa_answer_match_accepts_numeric_tolerance() -> None:
    result = finqa_answer_match(prediction="0.098641887", gold="0.09864")

    assert result.is_correct
    assert result.match_type == "numeric_tolerance"


def test_finqa_answer_match_rejects_numeric_value_outside_tolerance() -> None:
    result = finqa_answer_match(prediction="0.12", gold="0.09864")

    assert not result.is_correct
    assert result.match_type == "numeric_tolerance"


def test_finqa_answer_match_compares_yes_no_exactly() -> None:
    assert finqa_answer_match(prediction="YES", gold="yes").is_correct
    assert not finqa_answer_match(prediction="yes", gold="no").is_correct


def test_finqa_answer_match_can_normalize_percent_prediction() -> None:
    result = finqa_answer_match(prediction="9.864%", gold="0.09864")

    assert result.is_correct
    assert result.note == "Converted percent prediction to decimal."


def test_finqa_answer_match_handles_missing_prediction() -> None:
    result = finqa_answer_match(prediction=None, gold="94.0")

    assert not result.is_correct
    assert result.match_type == "missing_prediction"


def test_exact_match_api_remains_available() -> None:
    result = exact_match(prediction="YES", gold="yes")

    assert result.is_correct
    assert result.match_type == "exact"


def test_evaluate_prediction_answer_matches_real_test_json_example() -> None:
    example = load_finqa_examples("data/raw/test.json")[0]
    prediction = build_prediction(
        selected_example=example,
        final_answer="94",
        model_output_text="subtract(5829, 5735)",
        parsed_output=None,
        execution_result=None,
        errors=[],
    )

    result = evaluate_prediction_answer(prediction, example)

    assert result.has_gold
    assert result.is_correct
    assert result.question == example.runtime.question
    assert result.gold_answer == "94.0"
    assert result.prediction_program == "subtract(5829, 5735)"
    assert result.gold_program == "subtract(5829, 5735)"
    assert result.gold_inds == example.gold.supporting_facts
    assert result.selected_evidence == {}
    assert result.retrieved_all_gold_inds is False
    assert result.match is not None
    assert result.match.match_type == "numeric_tolerance"


def test_evaluate_prediction_answer_reports_retrieved_all_gold_inds() -> None:
    example = load_finqa_examples("data/raw/test.json")[0]
    selected = [
        RetrievedEvidence(
            unit=EvidenceUnit(evidence_id=evidence_id, source="text", text=text),
            score=1.0,
            rank=index,
            selected=True,
        )
        for index, (evidence_id, text) in enumerate(example.gold.supporting_facts.items(), start=1)
    ]
    prediction = build_prediction(
        selected_example=example,
        final_answer="94",
        model_output_text="subtract(5829, 5735)",
        parsed_output=None,
        execution_result=None,
        selected_evidence=selected,
        errors=[],
    )

    result = evaluate_prediction_answer(prediction, example)

    assert result.gold_inds == example.gold.supporting_facts
    assert list(result.selected_evidence) == list(example.gold.supporting_facts)
    assert result.retrieved_all_gold_inds is True


def test_evaluate_prediction_answer_matches_real_yes_no_example() -> None:
    example = next(
        item for item in load_finqa_examples("data/raw/test.json") if item.gold.executable_answer == "yes"
    )
    prediction = build_prediction(
        selected_example=example,
        final_answer="YES",
        model_output_text="greater(286.61, 198.09)",
        parsed_output=None,
        execution_result=None,
        errors=[],
    )

    result = evaluate_prediction_answer(prediction, example)

    assert result.has_gold
    assert result.is_correct
    assert result.match is not None
    assert result.match.match_type == "yes_no"


def test_evaluate_prediction_answer_skips_missing_gold() -> None:
    example = FinQAExample(
        runtime=RuntimeInputs(
            example_id="private-1",
            filename=None,
            question="What is revenue?",
            pre_text=[],
            post_text=[],
            table=[],
        )
    )
    prediction = build_prediction(
        selected_example=example,
        final_answer="10",
        model_output_text="10",
        parsed_output=None,
        execution_result=None,
        errors=[],
    )

    result = evaluate_prediction_answer(prediction, example)

    assert not result.has_gold
    assert not result.is_correct
    assert result.match is None
    assert result.prediction_program == "10"
    assert result.gold_program is None


def test_evaluate_prediction_batch_summarizes_labeled_examples() -> None:
    examples = load_finqa_examples("data/raw/test.json")[:3]
    predictions = [
        build_prediction(
            selected_example=examples[0],
            final_answer="94",
            model_output_text="subtract(5829, 5735)",
            parsed_output=None,
            execution_result=None,
            errors=[],
        ),
        build_prediction(
            selected_example=examples[1],
            final_answer="999",
            model_output_text="999",
            parsed_output=None,
            execution_result=None,
            errors=[],
        ),
    ]

    report = evaluate_prediction_batch(predictions=predictions, examples=examples)

    assert report.summary.total_examples == 3
    assert report.summary.examples_with_gold == 3
    assert report.summary.examples_without_gold == 0
    assert report.summary.answered_examples == 2
    assert report.summary.unanswered_examples == 1
    assert report.summary.correct_answers == 1
    assert report.summary.accuracy == 1 / 3
    assert report.summary.missing_predictions == 1
    assert report.summary.scored
    assert len(report.details) == 3
    assert report.details[0].is_correct
    assert report.details[0].prediction_program == "subtract(5829, 5735)"
    assert report.details[0].gold_program == "subtract(5829, 5735)"
    assert report.details[2].prediction_errors == ("Missing prediction.",)
    assert report.details[2].gold_program is not None


def test_evaluate_prediction_batch_counts_pipeline_errors() -> None:
    example = load_finqa_examples("data/raw/test.json")[0]
    predictions = [
        build_prediction(
            selected_example=example,
            final_answer=None,
            model_output_text="The answer is 94.",
            parsed_output=None,
            execution_result=None,
            errors=[
                "Failed to parse model output.",
                "Failed to execute parsed output.",
                "Some other warning.",
            ],
        )
    ]

    report = evaluate_prediction_batch(predictions=predictions, examples=[example])

    assert report.summary.parse_failures == 1
    assert report.summary.execution_failures == 1
    assert report.summary.other_pipeline_errors == 1
    assert report.summary.correct_answers == 0


def test_evaluate_prediction_batch_supports_inference_only_examples() -> None:
    example = _private_example()
    prediction = build_prediction(
        selected_example=example,
        final_answer="10",
        model_output_text="10",
        parsed_output=None,
        execution_result=None,
        errors=[],
    )

    report = evaluate_prediction_batch(predictions=[prediction], examples=[example])

    assert report.summary.total_examples == 1
    assert report.summary.examples_with_gold == 0
    assert report.summary.examples_without_gold == 1
    assert report.summary.answered_examples == 1
    assert report.summary.correct_answers == 0
    assert report.summary.accuracy is None
    assert not report.summary.scored
    assert not report.details[0].has_gold


def test_write_batch_evaluation_outputs(tmp_path) -> None:
    examples = load_finqa_examples("data/raw/test.json")[:1]
    prediction = build_prediction(
        selected_example=examples[0],
        final_answer="94",
        model_output_text="subtract(5829, 5735)",
        parsed_output=None,
        execution_result=None,
        errors=[],
    )
    report = evaluate_prediction_batch(predictions=[prediction], examples=examples)

    paths = write_batch_evaluation_outputs(
        report,
        output_dir=tmp_path,
        run_name="sample_run",
    )

    assert set(paths) == {"summary", "details", "markdown"}
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    details = json.loads(paths["details"].read_text(encoding="utf-8"))
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert summary["correct_answers"] == 1
    assert details[0]["is_correct"] is True
    assert details[0]["question"] == examples[0].runtime.question
    assert "gold_inds" in details[0]
    assert "selected_evidence" in details[0]
    assert "retrieved_all_gold_inds" in details[0]
    assert details[0]["prediction_program"] == "subtract(5829, 5735)"
    assert details[0]["gold_program"] == "subtract(5829, 5735)"
    assert "# Answer Evaluation Details" in markdown
    assert "predicted_program" in markdown


def _private_example() -> FinQAExample:
    return FinQAExample(
        runtime=RuntimeInputs(
            example_id="private-1",
            filename=None,
            question="What is revenue?",
            pre_text=[],
            post_text=[],
            table=[],
        )
    )

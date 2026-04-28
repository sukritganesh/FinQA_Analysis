from __future__ import annotations

from src.data.loader import load_finqa_examples
from src.eval.answer import evaluate_prediction_answer
from src.graph.workflow import build_single_example_workflow, run_single_example_workflow
from src.llm.client import ModelConfig, ModelResponse
from src.retrieval.base import RetrievalConfig


def test_single_example_workflow_returns_prediction_for_one_example() -> None:
    example = load_finqa_examples("data/raw/test.json")[0]
    client = FakeModelClient(ModelResponse(text="subtract(5829, 5735)"))
    graph = build_single_example_workflow()

    result = graph.invoke(
        {
            "selected_example": example,
            "retrieval_config": RetrievalConfig(mode="by_source", top_k_text=2, top_k_table=2),
            "prompt_dir": "configs/prompts/finqa_prompt_B_compact",
            "model_config": ModelConfig(model="qwen14b-awq"),
            "model_client": client,
            "errors": [],
        }
    )

    prediction = result["prediction"]
    evaluation = evaluate_prediction_answer(prediction, example)

    assert result["errors"] == []
    assert result["question"] == example.runtime.question
    assert result["evidence_units"]
    assert result["retrieved_evidence"]
    assert example.runtime.question in result["prompt"]
    assert client.seen_prompt is not None
    assert prediction.answer == "94"
    assert prediction.model_output_text == "subtract(5829, 5735)"
    assert evaluation.is_correct


def test_run_single_example_workflow_helper_returns_prediction() -> None:
    example = load_finqa_examples("data/raw/test.json")[0]
    client = FakeModelClient(ModelResponse(text="subtract(5829, 5735)"))

    prediction = run_single_example_workflow(
        example,
        retrieval_config=RetrievalConfig(mode="combined", top_k=4),
        prompt_dir="configs/prompts/finqa_prompt_B_compact",
        model_client=client,
    )

    assert prediction.example_id == example.runtime.example_id
    assert prediction.answer == "94"
    assert prediction.is_successful


def test_single_example_workflow_formats_prediction_when_model_output_is_bad() -> None:
    example = load_finqa_examples("data/raw/test.json")[0]
    client = FakeModelClient(ModelResponse(text="The answer is 94."))

    prediction = run_single_example_workflow(
        example,
        retrieval_config=RetrievalConfig(mode="combined", top_k=2),
        prompt_dir="configs/prompts/finqa_prompt_B_compact",
        model_client=client,
    )

    assert prediction.answer is None
    assert prediction.model_output_text == "The answer is 94."
    assert any("Failed to parse model output" in error for error in prediction.errors)
    assert any("Missing final_answer" in error for error in prediction.errors)


class FakeModelClient:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.seen_prompt: str | None = None
        self.seen_config: ModelConfig | None = None

    def generate(self, prompt: str, config: ModelConfig | None = None) -> ModelResponse:
        self.seen_prompt = prompt
        self.seen_config = config
        return self.response

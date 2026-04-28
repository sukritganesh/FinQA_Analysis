from __future__ import annotations

import json

from src.data.loader import load_finqa_examples
from src.eval.prediction import build_prediction
from src.llm.client import OpenAIClient, VLLMClient

from scripts import run_pipeline


def test_load_pipeline_config_reads_sample_yaml() -> None:
    config = run_pipeline.load_pipeline_config("configs/runs/dev_30_qwen14b_awq.yaml")

    assert config.input_path.as_posix() == "data/raw/dev.json"
    assert config.limit == 30
    assert config.retrieval_config.mode == "by_source"
    assert config.retrieval_config.top_k_text == 3
    assert config.prompt_dir is not None
    assert config.prompt_dir.as_posix() == "configs/prompts/finqa_prompt_B_compact"
    assert config.model_config.resolved_model == "Qwen/Qwen2.5-14B-Instruct-AWQ"
    assert config.run_name == "dev_30_qwen14b_awq"


def test_load_pipeline_config_reads_openai_sample_yaml() -> None:
    config = run_pipeline.load_pipeline_config("configs/runs/train_100_openai.yaml")

    assert config.input_path.as_posix() == "data/processed/train_100.json"
    assert config.limit == 100
    assert config.model_config.provider == "openai"
    assert config.model_config.model == "gpt-4.1-mini"
    assert config.model_config.base_url == "https://api.openai.com/v1"
    assert config.model_config.api_key is None
    assert config.model_config.api_key_env == "OPENAI_API_KEY"
    assert config.run_name == "train_100_openai"


def test_build_model_client_selects_provider_adapter() -> None:
    vllm_config = run_pipeline.load_pipeline_config("configs/runs/dev_30_qwen14b_awq.yaml")
    openai_config = run_pipeline.load_pipeline_config("configs/runs/train_100_openai.yaml")

    assert isinstance(run_pipeline.build_model_client(vllm_config.model_config), VLLMClient)
    assert isinstance(run_pipeline.build_model_client(openai_config.model_config), OpenAIClient)


def test_write_prediction_file_writes_compact_serializable_records(tmp_path) -> None:
    example = load_finqa_examples("data/raw/test.json")[0]
    prediction = build_prediction(
        selected_example=example,
        final_answer="94",
        model_output_text="subtract(5829, 5735)",
        parsed_output=None,
        execution_result=None,
        errors=[],
    )
    output_path = tmp_path / "predictions.json"

    run_pipeline.write_prediction_file(output_path, [prediction])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == [
        {
            "example_id": example.runtime.example_id,
            "question": example.runtime.question,
            "answer": "94",
            "normalized_answer": "94",
            "model_output_text": "subtract(5829, 5735)",
            "errors": [],
        }
    ]


def test_run_pipeline_from_config_uses_workflow_and_writes_outputs(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "run.yaml"
    prediction_path = tmp_path / "predictions.json"
    evaluation_dir = tmp_path / "evaluation"
    config_path.write_text(
        f"""
data:
  input_path: data/raw/test.json
  limit: 1
retrieval:
  strategy: bm25
  mode: combined
  top_k: 2
  top_k_text: 1
  top_k_table: 1
prompt:
  prompt_dir: configs/prompts/finqa_prompt_B_compact
model:
  model: qwen3b
outputs:
  run_name: fake_run
  prediction_path: {prediction_path.as_posix()}
  evaluation_dir: {evaluation_dir.as_posix()}
  evaluate: true
runtime:
  continue_on_error: true
""",
        encoding="utf-8",
    )

    def fake_workflow(example, **kwargs):
        return build_prediction(
            selected_example=example,
            final_answer="94",
            model_output_text="subtract(5829, 5735)",
            parsed_output=None,
            execution_result=None,
            errors=[],
        )

    monkeypatch.setattr(run_pipeline, "run_single_example_workflow", fake_workflow)

    config = run_pipeline.load_pipeline_config(config_path)
    report = run_pipeline.run_pipeline_from_config(config)

    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    summary = json.loads((evaluation_dir / "fake_run_summary.json").read_text(encoding="utf-8"))
    archived_config = evaluation_dir / "test_configuration.yaml"
    assert report is not None
    assert predictions[0]["answer"] == "94"
    assert summary["correct_answers"] == 1
    assert archived_config.read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")


def test_run_pipeline_from_config_stops_on_openai_quota_error(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "run.yaml"
    prediction_path = tmp_path / "predictions.json"
    config_path.write_text(
        f"""
data:
  input_path: data/raw/test.json
  limit: 1
retrieval:
  strategy: bm25
  mode: combined
  top_k: 2
  top_k_text: 1
  top_k_table: 1
prompt:
  prompt_dir: configs/prompts/finqa_prompt_B_compact
model:
  provider: openai
  model: gpt-4.1-mini
  api_key_env: OPENAI_API_KEY
outputs:
  run_name: fake_openai_run
  prediction_path: {prediction_path.as_posix()}
  evaluate: false
runtime:
  continue_on_error: true
""",
        encoding="utf-8",
    )

    def fake_workflow(example, **kwargs):
        return build_prediction(
            selected_example=example,
            final_answer=None,
            model_output_text=None,
            parsed_output=None,
            execution_result=None,
            errors=[
                "Failed to call model: OpenAI request failed with HTTP 429: "
                '{"error":{"code":"insufficient_quota"}}'
            ],
        )

    monkeypatch.setattr(run_pipeline, "run_single_example_workflow", fake_workflow)

    config = run_pipeline.load_pipeline_config(config_path)
    try:
        run_pipeline.run_pipeline_from_config(config)
    except RuntimeError as exc:
        assert "OpenAI provider call failed" in str(exc)
        assert "insufficient_quota" in str(exc)
    else:
        raise AssertionError("Expected OpenAI quota errors to stop the batch run.")

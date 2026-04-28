"""Run the FinQA end-to-end pipeline from a YAML config."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_finqa_examples
from src.data.schemas import FinQAExample
from src.eval.answer import evaluate_prediction_batch, write_batch_evaluation_outputs
from src.eval.prediction import Prediction, build_prediction
from src.graph.workflow import run_single_example_workflow
from src.llm.client import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_VLLM_BASE_URL,
    DEFAULT_VLLM_MODEL,
    ModelClient,
    ModelConfig,
    OpenAIClient,
    ProviderName,
    VLLMClient,
)
from src.retrieval.base import RetrievalConfig


@dataclass(frozen=True, slots=True)
class PipelineRunConfig:
    """Config for one batch pipeline run."""

    config_path: Path
    input_path: Path
    run_name: str
    retrieval_config: RetrievalConfig
    model_config: ModelConfig
    prompt_dir: Path | None
    prediction_path: Path
    evaluation_dir: Path | None
    limit: int | None = None
    offset: int = 0
    evaluate: bool = True
    continue_on_error: bool = True
    abort_on_provider_error: bool = True


def main() -> None:
    args = build_parser().parse_args()
    config = load_pipeline_config(args.config)
    report = run_pipeline_from_config(config)

    print(f"Wrote predictions to {config.prediction_path}")
    if report is not None:
        accuracy = "n/a" if report.summary.accuracy is None else f"{report.summary.accuracy:.4f}"
        print(
            "Evaluation: "
            f"{report.summary.correct_answers}/{report.summary.examples_with_gold} correct, "
            f"accuracy={accuracy}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="Path to a YAML pipeline run config.",
    )
    return parser


def load_pipeline_config(path: str | Path) -> PipelineRunConfig:
    """Load a pipeline run config from YAML."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Pipeline config must be a YAML mapping.")

    data = _section(payload, "data")
    retrieval = _section(payload, "retrieval")
    prompt = payload.get("prompt") or {}
    model = _section(payload, "model")
    outputs = _section(payload, "outputs")
    runtime = payload.get("runtime") or {}

    run_name = str(outputs.get("run_name", config_path.stem))
    prediction_path = Path(
        outputs.get("prediction_path")
        or Path(outputs.get("prediction_dir", "reports/predictions")) / f"{run_name}_predictions.json"
    )
    evaluation_dir = outputs.get("evaluation_dir")

    return PipelineRunConfig(
        config_path=config_path,
        input_path=Path(_required(data, "input_path")),
        limit=_optional_int(data.get("limit")),
        offset=int(data.get("offset", 0)),
        retrieval_config=RetrievalConfig(
            strategy=str(retrieval.get("strategy", "bm25")),
            mode=str(retrieval.get("mode", "by_source")),
            top_k=int(retrieval.get("top_k", 5)),
            top_k_text=int(retrieval.get("top_k_text", 3)),
            top_k_table=int(retrieval.get("top_k_table", 3)),
        ),
        prompt_dir=Path(prompt["prompt_dir"]) if prompt.get("prompt_dir") else None,
        model_config=_load_model_config(model),
        run_name=run_name,
        prediction_path=prediction_path,
        evaluation_dir=Path(evaluation_dir) if evaluation_dir else None,
        evaluate=bool(outputs.get("evaluate", True)),
        continue_on_error=bool(runtime.get("continue_on_error", True)),
        abort_on_provider_error=bool(runtime.get("abort_on_provider_error", True)),
    )


def _load_model_config(model: dict[str, Any]) -> ModelConfig:
    """Load model-provider settings from a YAML model section."""
    provider = str(model.get("provider", "vllm")).lower()
    if provider not in {"vllm", "openai"}:
        raise ValueError("model.provider must be either 'vllm' or 'openai'.")

    provider_name: ProviderName = "openai" if provider == "openai" else "vllm"
    default_model = DEFAULT_OPENAI_MODEL if provider_name == "openai" else DEFAULT_VLLM_MODEL
    default_base_url = DEFAULT_OPENAI_BASE_URL if provider_name == "openai" else DEFAULT_VLLM_BASE_URL
    api_key = model.get("api_key")
    api_key_env = model.get("api_key_env")

    return ModelConfig(
        provider=provider_name,
        model=str(model.get("model", default_model)),
        base_url=str(model.get("base_url", default_base_url)),
        max_tokens=int(model.get("max_tokens", 256)),
        temperature=float(model.get("temperature", 0.0)),
        timeout_seconds=float(model.get("timeout_seconds", 60.0)),
        extra_body=dict(model.get("extra_body") or {}),
        api_key=str(api_key) if api_key is not None else None,
        api_key_env=str(api_key_env) if api_key_env is not None else None,
    )


def run_pipeline_from_config(config: PipelineRunConfig):
    """Run examples through the full pipeline and optionally evaluate them."""
    examples = _select_examples(load_finqa_examples(config.input_path), config.offset, config.limit)
    model_client = build_model_client(config.model_config)

    predictions: list[Prediction] = []
    for index, example in enumerate(examples):
        try:
            prediction = run_single_example_workflow(
                example,
                retrieval_config=config.retrieval_config,
                prompt_dir=config.prompt_dir,
                model_config=config.model_config,
                model_client=model_client,
            )
        except Exception as exc:  # noqa: BLE001
            if not config.continue_on_error:
                raise
            prediction = _failed_prediction(example, exc)
        predictions.append(prediction)
        print(f"[{index + 1}/{len(examples)}] {example.runtime.example_id}: {prediction.answer}")
        if config.abort_on_provider_error:
            _raise_for_provider_error(prediction, config.model_config)

    write_prediction_file(config.prediction_path, predictions)
    archive_run_config(config)

    if not config.evaluate:
        return None

    report = evaluate_prediction_batch(predictions=predictions, examples=examples)
    if config.evaluation_dir is not None:
        write_batch_evaluation_outputs(
            report,
            output_dir=config.evaluation_dir,
            run_name=config.run_name,
        )
    return report


def archive_run_config(config: PipelineRunConfig) -> Path:
    """Copy the YAML run config into the run output directory for reproducibility."""
    output_dir = config.evaluation_dir or config.prediction_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    archived_path = output_dir / "test_configuration.yaml"
    shutil.copyfile(config.config_path, archived_path)
    return archived_path


def build_model_client(config: ModelConfig) -> ModelClient:
    """Create the right model client for a configured provider."""
    if config.provider == "openai":
        return OpenAIClient(default_config=config)
    if config.provider == "vllm":
        return VLLMClient(default_config=config)
    raise ValueError(f"Unsupported model provider: {config.provider}")


def write_prediction_file(path: str | Path, predictions: list[Prediction]) -> None:
    """Write compact prediction records to JSON."""
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_prediction_record(prediction) for prediction in predictions]
    resolved_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prediction_record(prediction: Prediction) -> dict[str, Any]:
    return {
        "example_id": prediction.example_id,
        "question": prediction.question,
        "answer": prediction.answer,
        "normalized_answer": prediction.normalized_answer,
        "model_output_text": prediction.model_output_text,
        "errors": list(prediction.errors),
    }


def _failed_prediction(example: FinQAExample, exc: Exception) -> Prediction:
    return build_prediction(
        selected_example=example,
        final_answer=None,
        model_output_text=None,
        parsed_output=None,
        execution_result=None,
        errors=[f"Pipeline failed: {exc}"],
    )


def _raise_for_provider_error(prediction: Prediction, model_config: ModelConfig) -> None:
    """Stop a batch run when the configured model provider is unavailable."""
    if not prediction.errors:
        return

    joined_errors = "\n".join(prediction.errors)
    lower_errors = joined_errors.lower()
    provider = model_config.provider

    if provider == "openai" and _looks_like_openai_provider_error(lower_errors):
        raise RuntimeError(
            "OpenAI provider call failed before a model answer was produced. "
            "The run was stopped to avoid writing null predictions for the whole batch.\n\n"
            f"{joined_errors}"
        )


def _looks_like_openai_provider_error(lower_errors: str) -> bool:
    markers = (
        "insufficient_quota",
        "exceeded your current quota",
        "http 401",
        "http 403",
        "http 429",
        "invalid_api_key",
        "billing",
    )
    return "failed to call model" in lower_errors and any(marker in lower_errors for marker in markers)


def _select_examples(
    examples: list[FinQAExample],
    offset: int,
    limit: int | None,
) -> list[FinQAExample]:
    start = max(offset, 0)
    if limit is None:
        return examples[start:]
    return examples[start : start + limit]


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    section = payload.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Pipeline config must include a '{name}' mapping.")
    return section


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"Missing required config key: {key}")
    return payload[key]


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    main()

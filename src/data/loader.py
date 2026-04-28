"""Load and normalize FinQA examples from JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.schemas import ExampleMetadata, FinQAExample, GoldTargets, RuntimeInputs
from src.utils.io import read_json

_KNOWN_TOP_LEVEL_FIELDS = {
    "filename",
    "id",
    "post_text",
    "pre_text",
    "qa",
    "table",
    "table_ori",
    "table_retrieved",
    "table_retrieved_all",
    "text_retrieved",
    "text_retrieved_all",
}


def load_finqa_examples(path: str | Path) -> list[FinQAExample]:
    """Load a list of FinQA examples from a JSON file."""
    raw_data = read_json(path)
    if not isinstance(raw_data, list):
        msg = f"Expected top-level list in {path}, got {type(raw_data).__name__}"
        raise ValueError(msg)

    return [normalize_finqa_example(item) for item in raw_data]


def normalize_finqa_example(raw_item: dict[str, Any]) -> FinQAExample:
    """Normalize one raw FinQA item into the internal schema."""
    if not isinstance(raw_item, dict):
        msg = f"Expected each FinQA record to be an object, got {type(raw_item).__name__}"
        raise ValueError(msg)

    qa = raw_item.get("qa", {})
    if qa is None:
        qa = {}
    if not isinstance(qa, dict):
        msg = f"Expected qa field to be an object, got {type(qa).__name__}"
        raise ValueError(msg)

    runtime = RuntimeInputs(
        example_id=str(raw_item.get("id", "")),
        filename=_coerce_optional_string(raw_item.get("filename")),
        question=str(qa.get("question", "")),
        pre_text=_coerce_string_list(raw_item.get("pre_text")),
        post_text=_coerce_string_list(raw_item.get("post_text")),
        table=_coerce_table(raw_item.get("table")),
    )

    gold = GoldTargets(
        answer=_coerce_optional_string(qa.get("answer")),
        executable_answer=_coerce_optional_string(qa.get("exe_ans")),
        program=_coerce_optional_string(qa.get("program")),
        program_nested=_coerce_optional_string(qa.get("program_re")),
        supporting_facts=_coerce_supporting_facts(qa.get("gold_inds")),
        ann_text_rows=_coerce_int_list(qa.get("ann_text_rows")),
        ann_table_rows=_coerce_int_list(qa.get("ann_table_rows")),
        steps=_coerce_steps(qa.get("steps")),
        explanation=_coerce_optional_string(qa.get("explanation")),
    )

    metadata = ExampleMetadata(
        table_original=raw_item.get("table_ori"),
        model_input=qa.get("model_input"),
        tfidf_topn=qa.get("tfidftopn"),
        text_retrieved=raw_item.get("text_retrieved"),
        text_retrieved_all=raw_item.get("text_retrieved_all"),
        table_retrieved=raw_item.get("table_retrieved"),
        table_retrieved_all=raw_item.get("table_retrieved_all"),
        extra_fields=_collect_extra_fields(raw_item),
    )

    return FinQAExample(runtime=runtime, gold=gold, metadata=metadata)


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_table(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []

    rows: list[list[str]] = []
    for row in value:
        if isinstance(row, list):
            rows.append([str(cell) for cell in row])
        else:
            rows.append([str(row)])
    return rows


def _coerce_supporting_facts(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): str(val) for key, val in value.items()}
    if isinstance(value, list):
        return {str(index): str(item) for index, item in enumerate(value)}
    return {"value": str(value)}


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]

    ints: list[int] = []
    for item in value:
        try:
            ints.append(int(item))
        except (TypeError, ValueError):
            continue
    return ints


def _coerce_steps(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _collect_extra_fields(raw_item: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in raw_item.items()
        if key not in _KNOWN_TOP_LEVEL_FIELDS
    }

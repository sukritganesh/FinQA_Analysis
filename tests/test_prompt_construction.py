"""Tests for FinQA prompt construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.evidence import build_evidence_units
from src.data.loader import load_finqa_examples
from src.data.schemas import EvidenceUnit
from src.llm.prompts import (
    assemble_reasoning_prompt,
    build_langchain_prompt_template,
    build_langchain_reasoning_prompt,
    build_reasoning_prompt,
    format_evidence_context,
    load_prompt_assets,
)
from src.retrieval.base import RetrievalConfig, RetrievedEvidence
from src.retrieval.bm25 import BM25Retriever


def test_load_prompt_assets_reads_expected_sections() -> None:
    assets = load_prompt_assets()

    assert "Return exactly one line" in assets.system
    assert "[table_6]" in assets.evidence_instructions
    assert "<stub column name> the <row name> of <column name> is <cell value>" in (
        assets.evidence_instructions
    )
    assert "stub column name = ( square feet in millions )" in assets.evidence_instructions
    assert "This text evidence says the relevant table compares 2015 net revenue to 2014 net revenue." in (
        assets.evidence_instructions
    )
    assert "DO NOT use any table functions" in (
        assets.evidence_instructions
    )
    assert "greater(a, b)" in assets.operation_guide
    assert "Do not use table operations to retrieve a single cell value." in assets.operation_guide
    assert "examples:" in assets.few_shot_examples.lower()
    assert "subtract(300.0, 240.0), divide(#0, 240.0)" in assets.few_shot_examples
    assert "greater(625, 500)" in assets.few_shot_examples
    assert "{question}" in assets.task_template
    assert "{evidence_context}" in assets.task_template
    assert "Do not use table operations for single-cell lookup." in assets.task_template


def test_format_evidence_context_preserves_selected_order() -> None:
    table_unit = EvidenceUnit(
        evidence_id="table_2",
        source="table",
        text="category the 2024 revenue of amount is 100 ;",
    )
    text_unit = EvidenceUnit(
        evidence_id="text_0",
        source="text",
        text="Revenue increased during the year.",
    )
    selected = [
        RetrievedEvidence(unit=table_unit, score=4.2, rank=1, selected=True),
        RetrievedEvidence(unit=text_unit, score=2.1, rank=2, selected=True),
    ]

    evidence_context = format_evidence_context(selected)

    assert evidence_context.splitlines() == [
        "[table_2] category the 2024 revenue of amount is 100 ;",
        "[text_0] Revenue increased during the year.",
    ]


def test_assemble_reasoning_prompt_fills_template_without_json_contract() -> None:
    assets = load_prompt_assets()

    prompt = assemble_reasoning_prompt(
        question="What was the revenue change?",
        evidence_context="[table_1] year the 2024 revenue of amount is 100 ;",
        assets=assets,
    )

    assert "What was the revenue change?" in prompt
    assert "[table_1] year the 2024 revenue of amount is 100 ;" in prompt
    assert "copy those numbers directly into the program" in prompt
    assert "stub column name = ( square feet in millions )" in prompt
    assert "Examples:" in prompt
    assert "Do not nest operations." in prompt
    assert "{question}" not in prompt
    assert "{evidence_context}" not in prompt


def test_langchain_prompt_template_uses_expected_variables() -> None:
    template = build_langchain_prompt_template()

    assert set(template.input_variables) == {"question", "evidence_context"}

    prompt = template.format(
        question="What is the answer?",
        evidence_context="[text_0] The answer is 42.",
    )

    assert "What is the answer?" in prompt
    assert "[text_0] The answer is 42." in prompt
    assert "Table evidence follows this schema:" in prompt
    assert "For the actual task, return only the final Output line." in prompt


def test_load_prompt_assets_skips_missing_optional_files(tmp_path: Path) -> None:
    (tmp_path / "system.txt").write_text("Return exactly one line.", encoding="utf-8")
    (tmp_path / "task_template.txt").write_text(
        "Question: {question}\nEvidence: {evidence_context}",
        encoding="utf-8",
    )

    assets = load_prompt_assets(tmp_path)
    prompt = assemble_reasoning_prompt(
        question="What is revenue?",
        evidence_context="[text_0] Revenue was 10.",
        assets=assets,
    )

    assert assets.evidence_instructions == ""
    assert assets.operation_guide == ""
    assert assets.few_shot_examples == ""
    assert "What is revenue?" in prompt


def test_load_prompt_assets_uses_manifest_order_and_custom_filenames(tmp_path: Path) -> None:
    (tmp_path / "intro.txt").write_text("Intro section.", encoding="utf-8")
    (tmp_path / "task.txt").write_text("Question: {question}", encoding="utf-8")
    (tmp_path / "context.txt").write_text("Evidence:\n{evidence_context}", encoding="utf-8")
    (tmp_path / "prompt.yaml").write_text(
        "\n".join(
            [
                "sections:",
                "  - file: task.txt",
                "    name: task",
                "  - intro.txt",
                "  - file: context.txt",
                "    name: context",
            ]
        ),
        encoding="utf-8",
    )

    assets = load_prompt_assets(tmp_path)
    prompt = assemble_reasoning_prompt(
        question="What is revenue?",
        evidence_context="[text_0] Revenue was 10.",
        assets=assets,
    )

    assert [section.name for section in assets.sections] == ["task", "intro", "context"]
    assert prompt.index("Question: What is revenue?") < prompt.index("Intro section.")
    assert prompt.index("Intro section.") < prompt.index("Evidence:")


def test_load_prompt_assets_requires_question_and_evidence_variables(tmp_path: Path) -> None:
    (tmp_path / "only_question.txt").write_text("Question: {question}", encoding="utf-8")
    (tmp_path / "prompt.yaml").write_text("sections:\n  - only_question.txt\n", encoding="utf-8")

    with pytest.raises(ValueError, match="evidence_context"):
        load_prompt_assets(tmp_path)


def test_langchain_reasoning_prompt_matches_plain_python_builder() -> None:
    selected = [
        EvidenceUnit(
            evidence_id="table_1",
            source="table",
            text="year the 2024 revenue of amount is 100 ;",
        ),
        EvidenceUnit(
            evidence_id="table_2",
            source="table",
            text="year the 2023 revenue of amount is 90 ;",
        ),
    ]

    plain_prompt = build_reasoning_prompt(
        question="What was the revenue change?",
        selected_evidence=selected,
    )
    langchain_prompt = build_langchain_reasoning_prompt(
        question="What was the revenue change?",
        selected_evidence=selected,
    )

    assert langchain_prompt == plain_prompt


def test_build_reasoning_prompt_accepts_real_retrieved_evidence() -> None:
    example = load_finqa_examples("data/raw/test.json")[0]
    evidence_units = build_evidence_units(example)
    retrieval = BM25Retriever().retrieve(
        question=example.runtime.question,
        evidence_units=evidence_units,
        config=RetrievalConfig(mode="combined", top_k=3),
    )

    prompt = build_reasoning_prompt(
        question=example.runtime.question,
        selected_evidence=retrieval.selected_evidence,
    )

    assert example.runtime.question in prompt
    for item in retrieval.selected_evidence:
        assert f"[{item.unit.evidence_id}]" in prompt
        assert item.unit.text in prompt

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm.parser import ParseError, parse_reasoning_output


DATA_DIR = Path("data/raw")


def test_parse_direct_yes_no_answers() -> None:
    yes = parse_reasoning_output(" yes ")
    no = parse_reasoning_output("NO")

    assert yes.kind == "direct_answer"
    assert yes.answer == "yes"
    assert no.kind == "direct_answer"
    assert no.answer == "no"


def test_parse_direct_numeric_answer() -> None:
    parsed = parse_reasoning_output("$1,234.50")

    assert parsed.kind == "direct_answer"
    assert parsed.answer == "$1,234.50"
    assert parsed.steps == ()


def test_parse_single_operation_program() -> None:
    parsed = parse_reasoning_output("subtract(5829, 5735)")

    assert parsed.kind == "program"
    assert len(parsed.steps) == 1
    step = parsed.steps[0]
    assert step.index == 0
    assert step.operation == "subtract"
    assert [arg.raw for arg in step.arguments] == ["5829", "5735"]
    assert [arg.kind for arg in step.arguments] == ["number", "number"]


def test_parse_chained_program_with_reference() -> None:
    parsed = parse_reasoning_output("subtract(153.7, 139.9), divide(#0, 139.9)")

    assert len(parsed.steps) == 2
    assert parsed.steps[1].operation == "divide"
    assert parsed.steps[1].arguments[0].kind == "reference"
    assert parsed.steps[1].arguments[0].reference_index == 0


def test_parse_program_with_dataset_constants() -> None:
    parsed = parse_reasoning_output("subtract(193.5, const_100), divide(#0, const_100)")

    assert parsed.steps[0].arguments[1].kind == "constant"
    assert parsed.steps[0].arguments[1].raw == "const_100"
    assert parsed.steps[1].arguments[1].kind == "constant"


def test_parse_program_with_decimal_literals() -> None:
    parsed = parse_reasoning_output("multiply(0.50, 0.06), subtract(#0, .450)")

    assert parsed.steps[0].arguments[0].kind == "number"
    assert parsed.steps[0].arguments[1].kind == "number"
    assert parsed.steps[1].arguments[1].raw == ".450"


def test_parse_table_operation_program() -> None:
    parsed = parse_reasoning_output(
        "table_min(expected volatility, none), table_max(expected volatility, none), subtract(#1, #0)"
    )

    assert parsed.steps[0].operation == "table_min"
    assert parsed.steps[0].arguments[0].kind == "text"
    assert parsed.steps[0].arguments[0].raw == "expected volatility"
    assert parsed.steps[0].arguments[1].kind == "none"
    assert parsed.steps[2].operation == "subtract"


def test_parse_supported_gold_programs_from_test_json() -> None:
    programs = _sample_supported_gold_programs(DATA_DIR / "test.json", limit=20)

    assert len(programs) == 20
    for program in programs:
        parsed = parse_reasoning_output(program)
        assert parsed.kind == "program"
        assert len(parsed.steps) >= 1
        assert parsed.raw_text == program


def test_parse_supported_gold_programs_from_dev_json() -> None:
    programs = _sample_supported_gold_programs(DATA_DIR / "dev.json", limit=20)

    assert len(programs) == 20
    for program in programs:
        parsed = parse_reasoning_output(program)
        assert parsed.kind == "program"
        assert len(parsed.steps) >= 1


@pytest.mark.parametrize(
    "bad_output",
    [
        "",
        "The answer is subtract(5829, 5735).",
        '{"program": "subtract(5829, 5735)"}',
        "divide(subtract(153.7, 139.9), 139.9)",
        "subtract(1, 2), divide(#2, 3)",
        "divide(#0, 3)",
        "exp(1.1, 2)",
        "subtract(1)",
        "subtract(1, two)",
        "multiply(50%, 6%)",
        "50%",
        "table_sum(total obligations, 5)",
        "subtract(1, 2)\nsubtract(3, 4)",
    ],
)
def test_parse_rejects_bad_outputs(bad_output: str) -> None:
    with pytest.raises(ParseError):
        parse_reasoning_output(bad_output)


def _sample_supported_gold_programs(path: Path, limit: int) -> list[str]:
    supported_programs = []
    data = json.loads(path.read_text(encoding="utf-8"))
    for raw_example in data:
        program = raw_example.get("qa", {}).get("program")
        if not program:
            continue
        try:
            parse_reasoning_output(program)
        except ParseError:
            continue
        supported_programs.append(program)
        if len(supported_programs) >= limit:
            break
    return supported_programs

from __future__ import annotations

from decimal import Decimal

import pytest

from src.llm.parser import parse_reasoning_output
from src.tools.executor import ExecutionError, execute_parsed_output


def test_execute_direct_numeric_answer() -> None:
    parsed = parse_reasoning_output("$1,234.50")

    result = execute_parsed_output(parsed)

    assert result.final_answer == "1234.5"
    assert result.step_results == ()


def test_execute_direct_yes_answer() -> None:
    parsed = parse_reasoning_output("YES")

    result = execute_parsed_output(parsed)

    assert result.final_answer == "yes"


def test_execute_single_arithmetic_step() -> None:
    parsed = parse_reasoning_output("subtract(5829, 5735)")

    result = execute_parsed_output(parsed)

    assert result.final_answer == "94"
    assert len(result.step_results) == 1
    assert result.step_results[0].result == Decimal("94")


def test_execute_chained_program_with_reference() -> None:
    parsed = parse_reasoning_output("subtract(153.7, 139.9), divide(#0, 139.9)")

    result = execute_parsed_output(parsed)

    assert result.step_results[0].result == Decimal("13.8")
    assert result.step_results[1].raw_arguments == ("#0", "139.9")
    assert result.step_results[1].resolved_arguments[0] == Decimal("13.8")
    assert result.final_answer.startswith("0.098641")


def test_execute_program_with_dataset_constants() -> None:
    parsed = parse_reasoning_output("subtract(193.5, const_100), divide(#0, const_100)")

    result = execute_parsed_output(parsed)

    assert result.step_results[0].result == Decimal("93.5")
    assert result.final_answer == "0.935"


def test_execute_program_with_negative_dataset_constant() -> None:
    parsed = parse_reasoning_output("multiply(4.9, const_m1)")

    result = execute_parsed_output(parsed)

    assert result.final_answer == "-4.9"


def test_execute_greater_returns_yes_or_no() -> None:
    parsed = parse_reasoning_output("greater(286.61, 198.09)")

    result = execute_parsed_output(parsed)

    assert result.final_answer == "yes"
    assert result.step_results[0].result == "yes"


def test_execute_table_sum() -> None:
    table = [
        ["Metric", "2021", "2022", "2023"],
        ["total obligations", "10", "20", "30"],
    ]
    parsed = parse_reasoning_output("table_sum(total obligations, none)")

    result = execute_parsed_output(parsed, table=table)

    assert result.final_answer == "60"


def test_execute_table_average_max_min_chain() -> None:
    table = [
        ["Metric", "2021", "2022", "2023"],
        ["expected volatility", "0.40", "0.35", "0.30"],
    ]
    parsed = parse_reasoning_output(
        "table_min(expected volatility, none), table_max(expected volatility, none), subtract(#1, #0)"
    )

    result = execute_parsed_output(parsed, table=table)

    assert result.step_results[0].result == Decimal("0.30")
    assert result.step_results[1].result == Decimal("0.40")
    assert result.final_answer == "0.1"


def test_execute_table_average_extracts_percent_from_repeated_financial_cell() -> None:
    table = [
        ["Metric", "2005", "2004", "2003"],
        ["effective tax rate", "26% ( 26 % )", "28% ( 28 % )", "26% ( 26 % )"],
    ]
    parsed = parse_reasoning_output("table_average(effective tax rate, none)")

    result = execute_parsed_output(parsed, table=table)

    assert result.step_results[0].result == Decimal("0.2666666666666666666666666667")
    assert result.final_answer.startswith("0.266666")


def test_execute_table_average_treats_plain_percent_cells_as_rates() -> None:
    table = [
        ["Metric", "2005", "2004", "2003"],
        ["effective tax rate", "26%", "28%", "26%"],
    ]
    parsed = parse_reasoning_output("table_average(effective tax rate, none)")

    result = execute_parsed_output(parsed, table=table)

    assert result.step_results[0].result == Decimal("0.2666666666666666666666666667")


def test_execute_table_min_extracts_signed_value_from_repeated_financial_cell() -> None:
    table = [
        ["Metric", "2005", "2004", "2003"],
        ["research and development credit net", "-26 ( 26 )", "-5 ( 5 )", "-7 ( 7 )"],
    ]
    parsed = parse_reasoning_output("table_min(research and development credit net, none)")

    result = execute_parsed_output(parsed, table=table)

    assert result.step_results[0].result == Decimal("-26")
    assert result.final_answer == "-26"


def test_execute_table_operation_requires_table() -> None:
    parsed = parse_reasoning_output("table_sum(total obligations, none)")

    with pytest.raises(ExecutionError, match="requires table data"):
        execute_parsed_output(parsed)


def test_execute_table_operation_requires_matching_row() -> None:
    parsed = parse_reasoning_output("table_sum(total obligations, none)")

    with pytest.raises(ExecutionError, match="Could not find table row"):
        execute_parsed_output(parsed, table=[["Metric", "2021"], ["other row", "10"]])


def test_execute_table_operation_requires_numeric_values() -> None:
    parsed = parse_reasoning_output("table_sum(total obligations, none)")

    with pytest.raises(ExecutionError, match="did not contain numeric values"):
        execute_parsed_output(parsed, table=[["Metric", "2021"], ["total obligations", "n/a"]])


def test_execute_raises_when_referencing_yes_no_result() -> None:
    parsed = parse_reasoning_output("greater(2, 1), add(#0, 1)")

    with pytest.raises(ExecutionError, match="non-numeric"):
        execute_parsed_output(parsed)

"""Deterministic execution for parsed FinQA programs."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable

from src.llm.parser import ParsedReasoningOutput, ProgramArgument, ProgramStep, TABLE_OPERATIONS
from src.tools.calculator import CalculatorResult, apply_operation, format_decimal, parse_decimal


ExecutionValue = Decimal | str

_CONSTANT_RE = re.compile(r"const_(m?)([0-9]+)")


class ExecutionError(ValueError):
    """Raised when a parsed output cannot be executed deterministically."""


@dataclass(frozen=True, slots=True)
class ExecutionStepResult:
    """Result of one executed program step."""

    index: int
    operation: str
    raw_arguments: tuple[str, ...]
    resolved_arguments: tuple[ExecutionValue | None, ...]
    result: ExecutionValue


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Final answer plus optional trace from deterministic execution."""

    final_answer: str
    parsed_output: ParsedReasoningOutput
    step_results: tuple[ExecutionStepResult, ...] = field(default_factory=tuple)


def execute_parsed_output(
    parsed_output: ParsedReasoningOutput,
    table: list[list[str]] | None = None,
) -> ExecutionResult:
    """Execute a parsed direct answer or parsed linear program."""
    if parsed_output.kind == "direct_answer":
        if parsed_output.direct_answer is None:
            raise ExecutionError("Parsed direct answer is missing a value.")
        return ExecutionResult(
            final_answer=_format_direct_answer(parsed_output.direct_answer.value),
            parsed_output=parsed_output,
        )

    if parsed_output.program is None:
        raise ExecutionError("Parsed program is missing program steps.")

    step_results: list[ExecutionStepResult] = []
    values: list[ExecutionValue] = []
    for step in parsed_output.program.steps:
        result = _execute_step(step, values, table)
        step_results.append(result)
        values.append(result.result)

    if not step_results:
        raise ExecutionError("Program did not produce any step results.")

    return ExecutionResult(
        final_answer=_format_execution_value(step_results[-1].result),
        parsed_output=parsed_output,
        step_results=tuple(step_results),
    )


def _execute_step(
    step: ProgramStep,
    previous_values: list[ExecutionValue],
    table: list[list[str]] | None,
) -> ExecutionStepResult:
    if step.operation in TABLE_OPERATIONS:
        result = _execute_table_operation(step, table)
        resolved_arguments = (step.arguments[0].raw, None)
    else:
        resolved_arguments = tuple(
            _resolve_numeric_argument(argument, previous_values) for argument in step.arguments
        )
        result = apply_operation(step.operation, list(resolved_arguments))

    return ExecutionStepResult(
        index=step.index,
        operation=step.operation,
        raw_arguments=tuple(argument.raw for argument in step.arguments),
        resolved_arguments=resolved_arguments,
        result=result,
    )


def _resolve_numeric_argument(
    argument: ProgramArgument,
    previous_values: list[ExecutionValue],
) -> Decimal:
    if argument.kind == "number":
        return parse_decimal(argument.raw)

    if argument.kind == "constant":
        return _parse_constant(argument.raw)

    if argument.kind == "reference":
        reference_index = argument.reference_index
        if reference_index is None or reference_index >= len(previous_values):
            raise ExecutionError(f"Reference {argument.raw!r} is not available.")
        value = previous_values[reference_index]
        if isinstance(value, str):
            raise ExecutionError(f"Reference {argument.raw!r} points to non-numeric value {value!r}.")
        return value

    raise ExecutionError(f"Cannot resolve non-numeric argument {argument.raw!r}.")


def _parse_constant(raw: str) -> Decimal:
    match = _CONSTANT_RE.fullmatch(raw)
    if match is None:
        raise ExecutionError(f"Unsupported constant: {raw}")
    sign = -1 if match.group(1) == "m" else 1
    return Decimal(sign) * Decimal(match.group(2))


def _execute_table_operation(step: ProgramStep, table: list[list[str]] | None) -> Decimal:
    if table is None:
        raise ExecutionError(f"Table operation {step.operation!r} requires table data.")

    row_name = step.arguments[0].raw
    values = _find_numeric_table_row_values(table, row_name)

    if step.operation == "table_sum":
        return sum(values, start=Decimal("0"))
    if step.operation == "table_average":
        return sum(values, start=Decimal("0")) / Decimal(len(values))
    if step.operation == "table_max":
        return max(values)
    if step.operation == "table_min":
        return min(values)

    raise ExecutionError(f"Unsupported table operation: {step.operation}")


def _find_numeric_table_row_values(table: list[list[str]], row_name: str) -> list[Decimal]:
    normalized_target = _normalize_table_label(row_name)
    for row in table:
        if not row:
            continue
        if _normalize_table_label(row[0]) != normalized_target:
            continue

        values = list(_iter_numeric_values(row[1:]))
        if not values:
            raise ExecutionError(f"Table row {row_name!r} did not contain numeric values.")
        return values

    raise ExecutionError(f"Could not find table row matching {row_name!r}.")


def _iter_numeric_values(cells: Iterable[str]) -> Iterable[Decimal]:
    for cell in cells:
        try:
            yield parse_decimal(cell)
        except ValueError:
            continue


def _normalize_table_label(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    return " ".join(normalized.split())


def _format_direct_answer(value: str) -> str:
    if value in {"yes", "no"}:
        return value
    return format_decimal(parse_decimal(value))


def _format_execution_value(value: ExecutionValue) -> str:
    if isinstance(value, Decimal):
        return format_decimal(value)
    return value

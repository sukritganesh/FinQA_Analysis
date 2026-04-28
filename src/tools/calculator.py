"""Numeric normalization and simple deterministic execution helpers."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation


CalculatorResult = Decimal | str


def parse_decimal(value: str | int | float | Decimal) -> Decimal:
    """Parse a value into a Decimal after light financial-style cleanup."""
    if isinstance(value, Decimal):
        return value

    text = str(value).strip()
    negative = text.startswith("(") and text.endswith(")")
    cleaned = text.replace(",", "").replace("$", "").replace("%", "")
    cleaned = cleaned.strip("() ")

    try:
        number = Decimal(cleaned)
    except InvalidOperation as exc:
        msg = f"Could not parse decimal from value: {value!r}"
        raise ValueError(msg) from exc

    return -number if negative else number


def apply_operation(operation: str, operands: list[str | int | float | Decimal]) -> CalculatorResult:
    """Apply a simple deterministic numeric operation."""
    parsed = [parse_decimal(item) for item in operands]

    if operation == "add":
        return sum(parsed, start=Decimal("0"))
    if operation == "subtract":
        _require_operand_count(operation, parsed, expected=2)
        return parsed[0] - parsed[1]
    if operation == "multiply":
        result = Decimal("1")
        for value in parsed:
            result *= value
        return result
    if operation == "divide":
        _require_operand_count(operation, parsed, expected=2)
        if parsed[1] == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return parsed[0] / parsed[1]
    if operation == "greater":
        _require_operand_count(operation, parsed, expected=2)
        return "yes" if parsed[0] > parsed[1] else "no"

    msg = f"Unsupported operation: {operation}"
    raise ValueError(msg)


def format_decimal(value: Decimal, places: int | None = None) -> str:
    """Format a Decimal for readable output."""
    if places is not None:
        quantizer = Decimal("1").scaleb(-places)
        value = value.quantize(quantizer)

    normalized = value.normalize()
    return format(normalized, "f")


def _require_operand_count(operation: str, operands: list[Decimal], expected: int) -> None:
    if len(operands) != expected:
        msg = f"Operation {operation!r} expects {expected} operands, got {len(operands)}"
        raise ValueError(msg)

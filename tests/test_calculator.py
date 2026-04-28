from __future__ import annotations

from decimal import Decimal

import pytest

from src.tools.calculator import apply_operation, format_decimal, parse_decimal


def test_parse_decimal_handles_common_financial_formatting() -> None:
    assert parse_decimal("$1,234.50") == Decimal("1234.50")
    assert parse_decimal("(12)") == Decimal("-12")


def test_apply_operation_divide() -> None:
    result = apply_operation("divide", ["12", "3"])
    assert result == Decimal("4")


def test_apply_operation_greater_returns_yes_or_no() -> None:
    assert apply_operation("greater", ["12", "3"]) == "yes"
    assert apply_operation("greater", ["3", "12"]) == "no"


def test_apply_operation_raises_on_zero_division() -> None:
    with pytest.raises(ZeroDivisionError):
        apply_operation("divide", ["10", "0"])


def test_format_decimal_places() -> None:
    assert format_decimal(Decimal("1.234"), places=2) == "1.23"

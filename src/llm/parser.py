"""Parse one-line FinQA model outputs into typed objects."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


SUPPORTED_OPERATIONS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "greater",
        "table_sum",
        "table_average",
        "table_max",
        "table_min",
    }
)

TABLE_OPERATIONS = frozenset({"table_sum", "table_average", "table_max", "table_min"})
ARITHMETIC_OPERATIONS = SUPPORTED_OPERATIONS - TABLE_OPERATIONS

ArgumentKind = Literal["number", "reference", "constant", "none", "text"]
ParsedOutputKind = Literal["direct_answer", "program"]

_OPERATION_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_REFERENCE_RE = re.compile(r"#([0-9]+)")
_CONSTANT_RE = re.compile(r"const_m?[0-9]+")
_NUMBER_RE = re.compile(
    r"""
    \$?
    -?
    (?:
        (?:\d+(?:,\d{3})+|\d+)(?:\.\d*)?
        |
        \.\d+
    )
    """,
    re.VERBOSE,
)


class ParseError(ValueError):
    """Raised when a model output cannot be parsed safely."""


@dataclass(frozen=True, slots=True)
class ProgramArgument:
    """One raw argument to a parsed operation."""

    raw: str
    kind: ArgumentKind

    @property
    def reference_index(self) -> int | None:
        """Return the referenced step index when this is a `#n` argument."""
        if self.kind != "reference":
            return None
        return int(self.raw[1:])


@dataclass(frozen=True, slots=True)
class ProgramStep:
    """One operation in a linear FinQA program."""

    index: int
    operation: str
    arguments: tuple[ProgramArgument, ...]


@dataclass(frozen=True, slots=True)
class ParsedProgram:
    """A linear sequence of deterministic operation steps."""

    steps: tuple[ProgramStep, ...]


@dataclass(frozen=True, slots=True)
class DirectAnswer:
    """A direct final answer that does not require program execution."""

    value: str


@dataclass(frozen=True, slots=True)
class ParsedReasoningOutput:
    """Parsed model output for the V1 one-line FinQA contract."""

    kind: ParsedOutputKind
    direct_answer: DirectAnswer | None = None
    program: ParsedProgram | None = None
    raw_text: str = ""

    @property
    def answer(self) -> str | None:
        """Compatibility helper for older code that expected `.answer`."""
        if self.direct_answer is None:
            return None
        return self.direct_answer.value

    @property
    def steps(self) -> tuple[ProgramStep, ...]:
        """Return parsed program steps, or an empty tuple for direct answers."""
        if self.program is None:
            return ()
        return self.program.steps


def parse_reasoning_output(text: str) -> ParsedReasoningOutput:
    """Parse a direct answer or linear operation program from model output."""
    normalized = _normalize_output_text(text)
    direct_answer = _parse_direct_answer(normalized)
    if direct_answer is not None:
        return ParsedReasoningOutput(
            kind="direct_answer",
            direct_answer=direct_answer,
            raw_text=normalized,
        )

    program = _parse_program(normalized)
    return ParsedReasoningOutput(kind="program", program=program, raw_text=normalized)


def _normalize_output_text(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        raise ParseError("Model output is empty.")

    if "\n" in normalized or "\r" in normalized:
        raise ParseError("Model output must be exactly one line.")

    if normalized.startswith("{") or normalized.startswith("["):
        raise ParseError("Model output must not be JSON or markdown.")

    return normalized


def _parse_direct_answer(text: str) -> DirectAnswer | None:
    lowered = text.lower()
    if lowered in {"yes", "no"}:
        return DirectAnswer(value=lowered)

    if _NUMBER_RE.fullmatch(text):
        return DirectAnswer(value=text)

    return None


def _parse_program(text: str) -> ParsedProgram:
    steps: list[ProgramStep] = []
    position = 0
    length = len(text)

    while position < length:
        position = _skip_spaces(text, position)
        operation_match = _OPERATION_RE.match(text, position)
        if operation_match is None:
            raise ParseError(f"Expected operation name at position {position}.")

        operation = operation_match.group(0)
        if operation not in SUPPORTED_OPERATIONS:
            raise ParseError(f"Unsupported operation: {operation}")

        position = operation_match.end()
        if position >= length or text[position] != "(":
            raise ParseError(f"Expected '(' after operation {operation!r}.")

        close_position = _find_closing_parenthesis(text, position)
        arguments_text = text[position + 1 : close_position]
        arguments = tuple(_parse_arguments(arguments_text))
        step = ProgramStep(index=len(steps), operation=operation, arguments=arguments)
        _validate_step(step)
        steps.append(step)

        position = _skip_spaces(text, close_position + 1)
        if position == length:
            break
        if text[position] != ",":
            raise ParseError(f"Expected ',' between operations at position {position}.")
        position += 1

    if not steps:
        raise ParseError("Program did not contain any operation steps.")

    return ParsedProgram(steps=tuple(steps))


def _find_closing_parenthesis(text: str, open_position: int) -> int:
    position = open_position + 1
    while position < len(text):
        char = text[position]
        if char == "(":
            raise ParseError("Nested operation calls are not allowed.")
        if char == ")":
            return position
        position += 1

    raise ParseError("Missing closing ')' in operation call.")


def _parse_arguments(arguments_text: str) -> list[ProgramArgument]:
    raw_arguments = [argument.strip() for argument in arguments_text.split(",")]
    if any(argument == "" for argument in raw_arguments):
        raise ParseError("Operation arguments must not be empty.")
    return [_parse_argument(argument) for argument in raw_arguments]


def _parse_argument(raw: str) -> ProgramArgument:
    lowered = raw.lower()
    if lowered == "none":
        return ProgramArgument(raw=lowered, kind="none")

    if _REFERENCE_RE.fullmatch(raw):
        return ProgramArgument(raw=raw, kind="reference")

    if _CONSTANT_RE.fullmatch(raw):
        return ProgramArgument(raw=raw, kind="constant")

    if _NUMBER_RE.fullmatch(raw):
        return ProgramArgument(raw=raw, kind="number")

    if "(" in raw or ")" in raw:
        raise ParseError("Nested or malformed argument expression is not allowed.")

    return ProgramArgument(raw=raw, kind="text")


def _validate_step(step: ProgramStep) -> None:
    if len(step.arguments) != 2:
        raise ParseError(
            f"Operation {step.operation!r} expects 2 arguments, got {len(step.arguments)}."
        )

    if step.operation in TABLE_OPERATIONS:
        _validate_table_step(step)
    else:
        _validate_arithmetic_step(step)

    for argument in step.arguments:
        if argument.kind == "reference":
            reference_index = argument.reference_index
            if reference_index is None or reference_index >= step.index:
                raise ParseError(
                    f"Invalid reference {argument.raw!r} in step #{step.index}; "
                    "references must point to earlier steps."
                )


def _validate_arithmetic_step(step: ProgramStep) -> None:
    for argument in step.arguments:
        if argument.kind not in {"number", "reference", "constant"}:
            raise ParseError(
                f"Operation {step.operation!r} received unsupported argument {argument.raw!r}."
            )


def _validate_table_step(step: ProgramStep) -> None:
    row_name, marker = step.arguments
    if row_name.kind == "reference":
        raise ParseError(f"Table operation {step.operation!r} requires a row name.")
    if marker.kind != "none":
        raise ParseError(f"Table operation {step.operation!r} expects 'none' as argument 2.")


def _skip_spaces(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position

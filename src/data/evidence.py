"""Helpers for turning mixed document content into evidence units."""

from __future__ import annotations

import re

from src.data.schemas import EvidenceUnit, FinQAExample
from src.utils.text import normalize_whitespace

_TOKEN_END_PERIOD_RE = re.compile(r"(?<=[A-Za-z])\.(?=\s|$)")


def build_evidence_units(example: FinQAExample) -> list[EvidenceUnit]:
    """Build FinQA-aligned text and table evidence units for one example."""
    units: list[EvidenceUnit] = []
    units.extend(_build_text_units(example, start_index=0))
    units.extend(_build_table_units(example))
    return units


def _build_text_units(example: FinQAExample, start_index: int = 0) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    text_index = start_index

    for source_section, sections in (
        ("pre_text", example.runtime.pre_text),
        ("post_text", example.runtime.post_text),
    ):
        for source_index, text in enumerate(sections):
            cleaned = normalize_whitespace(text)
            if not cleaned:
                continue
            units.append(
                EvidenceUnit(
                    evidence_id=f"text_{text_index}",
                    source="text",
                    text=cleaned,
                    metadata={
                        "example_id": example.runtime.example_id,
                        "filename": example.runtime.filename,
                        "source_section": source_section,
                        "source_index": source_index,
                    },
                )
            )
            text_index += 1

    return units


def _build_table_units(example: FinQAExample) -> list[EvidenceUnit]:
    table = example.runtime.table
    if not table:
        return []

    header = table[0]
    units: list[EvidenceUnit] = []

    for row_index, row in enumerate(table):
        text = render_table_row(header, row)
        units.append(
            EvidenceUnit(
                evidence_id=f"table_{row_index}",
                source="table",
                text=text,
                metadata={
                    "example_id": example.runtime.example_id,
                    "filename": example.runtime.filename,
                    "source_section": "table",
                    "source_index": row_index,
                    "table_row_index": row_index,
                    "row_name": _get_cell(row, 0),
                    "stub_column_name": _get_cell(header, 0),
                    "column_names": header[1:],
                    "cell_values": row[1:],
                    "is_header_row": row_index == 0,
                },
            )
        )

    return units


def render_table_row(header: list[str], row: list[str]) -> str:
    """Render one table row using the FinQA gold-support style."""
    stub_column_name = _normalize_table_cell(_get_cell(header, 0))
    row_name = _normalize_table_cell(_get_cell(row, 0))
    parts: list[str] = []

    for column_index in range(1, max(len(header), len(row))):
        column_name = _normalize_table_cell(_get_cell(header, column_index))
        cell_value = _normalize_table_cell(_get_cell(row, column_index))
        if not column_name and not cell_value and not row_name:
            continue
        row_text = f" {row_name}" if row_name else ""
        column_text = f" {column_name}" if column_name else ""
        value_text = f" {cell_value}" if cell_value else ""
        parts.append(f"the{row_text} of{column_text} is{value_text} ;")

    rendered = " ".join(parts)
    if stub_column_name and rendered:
        return f"{stub_column_name} {rendered}"
    if rendered:
        return rendered
    return row_name


def _get_cell(row: list[str], index: int) -> str:
    if index >= len(row):
        return ""
    return str(row[index])


def _normalize_table_cell(value: str) -> str:
    """Normalize table text toward the style used in FinQA gold support."""
    value = value.replace(",", " ,").replace(":", " :")
    value = _TOKEN_END_PERIOD_RE.sub(" .", value)
    return normalize_whitespace(value)

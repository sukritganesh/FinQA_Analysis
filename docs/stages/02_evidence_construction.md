# 02. Evidence Construction

## Purpose

Convert normalized FinQA examples into retrieval-ready evidence units. This stage creates candidates; it does not decide which candidates are relevant.

## What This Stage Does

- Converts `pre_text` and `post_text` into text evidence units.
- Converts table rows into text-like evidence units.
- Assigns FinQA-aligned evidence ids such as `text_0` and `table_3`.
- Adds compact metadata so each unit can be traced back to its source.

## Key Files

- `src/data/evidence.py`
- `src/data/schemas.py`
- `src/graph/evidence.py`
- `tests/test_evidence.py`
- `tests/test_evidence_graph.py`

## Evidence Unit Shape

Each `EvidenceUnit` contains:

- `evidence_id`: stable id such as `text_0` or `table_3`
- `source`: `text` or `table`
- `text`: retrieval and prompt text
- `metadata`: source section, source index, row index, row name, column names, and related debug details

The metadata is intentionally small. The full raw example is available elsewhere.

## Text Evidence

FinQA already stores `pre_text` and `post_text` as sentence-like lists. The project preserves those items directly instead of re-splitting them.

Text ids continue across both sections:

```text
pre_text[0]  -> text_0
pre_text[1]  -> text_1
post_text[0] -> text_2
```

This preserves alignment with dataset support labels.

## Table Evidence

Each table row is rendered into the FinQA support style:

```text
<stub> the <row name> of <column name> is <cell value> ;
```

Example:

```text
the 2015 net revenue of amount ( in millions ) is $ 5829 ;
```

Wider rows contain one phrase per cell, separated by semicolons:

```text
company the american express of payments volume ( billions ) is 637 ; the american express of total volume ( billions ) is 647 ;
```

The header row is included as `table_0` so ids align with FinQA support annotations.

## Design Decisions

- Build evidence before retrieval so retrievers never parse raw FinQA JSON.
- Preserve original text items for support-id alignment.
- Render tables close to FinQA's gold support strings for easier debugging and evaluation.
- Keep source metadata available for retrieval analysis.

## Graph Integration

The graph node expects one selected example and returns evidence units:

```text
START -> build_evidence -> END
```

## Verification

Tests cover text id alignment, table rendering, metadata, and real FinQA support matching.

Observed table-support alignment:

```text
train: 7512 / 7512 matched
dev:   1071 / 1071 matched
test:  1318 / 1318 matched
```

## Notes

Future retrieval experiments can use richer rendering, but the current format is simple, inspectable, and aligned with FinQA labels.

# 03. Retrieval

## Purpose

Select the most relevant evidence units for a FinQA question using an explainable baseline. Retrieval narrows the context before prompting the model.

## What This Stage Does

- Scores all evidence units for one question.
- Returns a full ranked list for inspection.
- Returns a selected subset for prompt generation.
- Supports different retrieval strategies behind a small interface.

## Key Files

- `src/retrieval/base.py`
- `src/retrieval/bm25.py`
- `src/retrieval/factory.py`
- `src/graph/retrieval.py`
- `src/eval/retrieval.py`
- `scripts/evaluate_retrieval.py`
- `tests/test_bm25_retriever.py`
- `tests/test_retrieval_graph.py`
- `tests/test_retrieval_eval.py`

## Current Strategy

The implemented retriever is BM25 over evidence-unit text. Details are documented in [BM25 Retrieval](../details/bm25_retrieval.md).

Two selection modes are supported:

- `combined`: rank all evidence in one pool and select `top_k`.
- `by_source`: rank text and table evidence separately, then select `top_k_text` and `top_k_table`.

`by_source` is useful for FinQA because many questions need both narrative text and table rows.

Future retrieval techniques should be added behind the same `Retriever` interface so the rest of the pipeline can stay unchanged.

## Runtime Objects

- `RetrievalConfig`: strategy, mode, and top-k settings.
- `RetrievedEvidence`: one scored evidence unit with rank, optional source rank, and selection flag.
- `RetrievalResult`: full ranked evidence plus selected evidence.
- `Retriever`: protocol for swappable retrievers.

## Boundaries

Retrieval accepts:

- question
- evidence units
- retrieval config

Retrieval does not:

- load FinQA JSON
- build evidence
- format prompts
- call the model
- use gold labels during inference

## Graph Integration

The retrieval node expects `question`, `evidence_units`, and an optional `retrieval_config`:

```text
START -> retrieve_evidence -> END
```

It is also used inside the full workflow after evidence construction.

## Evaluation

Retrieval can be evaluated against FinQA gold support ids. Feel free to change the strategy, mode, and top-k parameters when comparing techniques.

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/test.json --limit 25 --mode by_source --top-k-text 3 --top-k-table 3 --log-path reports/retrieval/retrieval_eval_test_25.md
```

The evaluation output helps identify whether downstream answer failures start with missing evidence.

View the appendix for additional commands.

## Verification

Tests cover:

- BM25 tokenization
- combined and by-source ranking
- scoring all evidence before selection
- real FinQA retrieval smoke checks
- gold support overlap
- Markdown miss logging
- LangGraph integration

## Possible Improvements

- Add conservative boosts for year overlap, metric names, comparison terms, and percentage cues.
- Add hybrid BM25/vector retrieval if needed.
- Add a reranker only after the BM25 baseline is well understood.
- Keep strategy-specific implementation notes in `docs/details/` as new retrieval methods are added.

## Appendix

These commands are useful for comparing retrieval settings. The `--include-hits` runs are best for debugging because the Markdown output shows which selected evidence matched gold support ids.

### Sample Test Runs

Balanced retrieval on 100 test examples, selecting 3 text and 3 table candidates:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/test.json --limit 100 --mode by_source --top-k-text 3 --top-k-table 3 --include-hits --log-path reports/retrieval/retrieval_eval_test_100_by_source_3_3_with_hits.md
```

Balanced retrieval on 100 test examples, selecting 2 text and 2 table candidates:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/test.json --limit 100 --mode by_source --top-k-text 2 --top-k-table 2 --include-hits --log-path reports/retrieval/retrieval_eval_test_100_by_source_2_2_with_hits.md
```

Combined retrieval on 100 test examples, selecting the top 3 overall candidates:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/test.json --limit 100 --mode combined --top-k 3 --include-hits --log-path reports/retrieval/retrieval_eval_test_100_combined_top3_with_hits.md
```

Combined retrieval on 100 test examples, selecting the top 4 overall candidates:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/test.json --limit 100 --mode combined --top-k 4 --include-hits --log-path reports/retrieval/retrieval_eval_test_100_combined_top4_with_hits.md
```

Combined retrieval on 100 test examples, selecting the top 6 overall candidates:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/test.json --limit 100 --mode combined --top-k 6 --include-hits --log-path reports/retrieval/retrieval_eval_test_100_combined_top6_with_hits.md
```

### Full-Split Runs

Train split, combined top 3:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/train.json --mode combined --top-k 3 --log-path reports/retrieval/retrieval_eval_train_all_combined_top3.md
```

Train split, balanced 2 text and 2 table candidates:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/train.json --mode by_source --top-k-text 2 --top-k-table 2 --log-path reports/retrieval/retrieval_eval_train_all_by_source_2_2.md
```

Dev split, combined top 3:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/dev.json --mode combined --top-k 3 --log-path reports/retrieval/retrieval_eval_dev_all_combined_top3.md
```

Dev split, balanced 2 text and 2 table candidates:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/dev.json --mode by_source --top-k-text 2 --top-k-table 2 --log-path reports/retrieval/retrieval_eval_dev_all_by_source_2_2.md
```

Test split, combined top 3:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/test.json --mode combined --top-k 3 --log-path reports/retrieval/retrieval_eval_test_all_combined_top3.md
```

Test split, balanced 2 text and 2 table candidates:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/test.json --mode by_source --top-k-text 2 --top-k-table 2 --log-path reports/retrieval/retrieval_eval_test_all_by_source_2_2.md
```

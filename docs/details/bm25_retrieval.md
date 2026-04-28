# BM25 Retrieval

This document explains the BM25 retrieval strategy used by the FinQA pipeline.

BM25 is the current retrieval baseline because it is simple, fast, offline, and easy to inspect. That matters for FinQA because many questions depend on exact overlap in years, metric names, company terms, and table labels.

## Where It Fits

The retrieval stage receives:

- one question
- evidence units for the same FinQA example
- a `RetrievalConfig`

BM25 scores the evidence units and returns:

- all scored evidence for inspection
- a selected subset for prompt generation

The retriever does not load data, build evidence, call the model, or use gold labels during inference.

## Key Files

- `src/retrieval/bm25.py`
- `src/retrieval/base.py`
- `src/retrieval/factory.py`
- `src/graph/retrieval.py`
- `src/eval/retrieval.py`
- `scripts/evaluate_retrieval.py`
- `tests/test_bm25_retriever.py`

## Tokenization

The BM25 retriever uses lightweight tokenization:

```python
normalize_for_matching(text).split()
```

This means the same normalization style is applied to questions and evidence text before scoring.

The goal is not linguistic sophistication. The goal is a transparent baseline that is easy to debug when a relevant row or sentence is missed.

## Scoring

The implementation uses `BM25Okapi` from `rank-bm25`.

For each pool of evidence:

1. tokenize every evidence unit
2. tokenize the question
3. compute one BM25 score per evidence unit
4. sort by descending score
5. break score ties by evidence id for stable output

Each returned item is represented as `RetrievedEvidence` with:

- `unit`
- `score`
- `rank`
- `source_rank`
- `selected`
- `metadata`

The metadata includes the retrieval strategy:

```python
{"retrieval_strategy": "bm25"}
```

## Selection Modes

BM25 supports two selection modes.

### Combined

`combined` mode scores all evidence units in one pool and selects the top `top_k` items overall.

Use this when you want the retriever to choose freely between text and table evidence.

Example config:

```yaml
retrieval:
  strategy: bm25
  mode: combined
  top_k: 6
```

### By Source

`by_source` mode scores text evidence and table evidence in separate pools.

It selects:

- `top_k_text` text items
- `top_k_table` table items

Use this when you want to guarantee that both narrative evidence and table evidence have space in the prompt.

Example config:

```yaml
retrieval:
  strategy: bm25
  mode: by_source
  top_k_text: 3
  top_k_table: 3
```

This is the default mode for many FinQA runs because questions often need both explanatory text and exact table values.

## Evaluation

BM25 retrieval can be evaluated against FinQA gold support ids.

Quick test run:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/test.json --limit 25 --mode by_source --top-k-text 3 --top-k-table 3 --log-path reports/retrieval/retrieval_eval_test_25.md
```

Debug run with hit details:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/test.json --limit 100 --mode by_source --top-k-text 3 --top-k-table 3 --include-hits --log-path reports/retrieval/retrieval_eval_test_100_by_source_3_3_with_hits.md
```

Combined-mode comparison:

```bash
.venv/bin/python scripts/evaluate_retrieval.py --input data/raw/test.json --limit 100 --mode combined --top-k 6 --include-hits --log-path reports/retrieval/retrieval_eval_test_100_combined_top6_with_hits.md
```

The Markdown logs are useful for checking whether selected evidence overlaps with `qa.gold_inds`.

## Strengths

- Easy to explain in the report.
- Runs locally without model calls.
- Works well when question terms overlap with row labels, years, and financial metrics.
- Produces inspectable scores and rankings.
- Provides a strong baseline before adding embeddings or reranking.

## Limitations

- Can miss evidence when the question uses different wording from the report.
- Does not understand arithmetic intent.
- Does not know that some terms are more important, such as years or metric names.
- Scores each evidence unit independently.
- Does not use semantic similarity.

## Likely Improvements

Good next steps:

- Add year-overlap boosts.
- Add metric-name or keyword boosts.
- Add comparison cue handling for questions about increase, decrease, difference, or percentage change.
- Try hybrid BM25 plus vector retrieval.
- Add reranking over a small BM25 candidate set.

These should be added only after the BM25 baseline is measured and documented.

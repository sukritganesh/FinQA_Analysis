"""Evaluation helpers for evidence retrieval quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from src.data.evidence import build_evidence_units
from src.data.loader import load_finqa_examples
from src.data.schemas import FinQAExample
from src.retrieval.base import RetrievalConfig, Retriever
from src.retrieval.factory import build_retriever


@dataclass(slots=True)
class RetrievalEvaluationSummary:
    """Summary of retrieval overlap against FinQA gold support ids."""

    total_examples: int
    examples_with_gold: int
    examples_with_hit: int
    recall_at_selection: float


@dataclass(slots=True)
class RetrievalExampleResult:
    """Per-example retrieval evaluation details."""

    example_id: str
    question: str
    gold_ids: list[str]
    selected_ids: list[str]
    hit: bool
    missing_gold_ids: list[str]
    gold_ranks: dict[str, int | None]
    selected_evidence_preview: list[str]
    missing_gold_evidence_preview: list[str]


@dataclass(slots=True)
class RetrievalEvaluationReport:
    """Detailed retrieval evaluation report."""

    summary: RetrievalEvaluationSummary
    details: list[RetrievalExampleResult]
    config: RetrievalConfig
    data_path: str | None = None

    def to_markdown(self, include_hits: bool = False) -> str:
        """Render a readable Markdown report for local inspection."""
        lines = [
            "# Retrieval Evaluation Report",
            "",
            "## Summary",
            "",
            f"- data_path: {self.data_path or 'in-memory examples'}",
            f"- config: `{asdict(self.config)}`",
            f"- total_examples: {self.summary.total_examples}",
            f"- examples_with_gold: {self.summary.examples_with_gold}",
            f"- examples_with_hit: {self.summary.examples_with_hit}",
            f"- recall_at_selection: {self.summary.recall_at_selection:.4f}",
            "",
            "## Misses",
            "",
        ]

        misses = [detail for detail in self.details if not detail.hit]
        if not misses:
            lines.append("No misses in this run.")
            lines.append("")

        for detail in misses:
            lines.extend(
                [
                    f"### MISS: {detail.example_id}",
                    "",
                    f"Question: {detail.question}",
                    "",
                    f"Gold ids: `{detail.gold_ids}`",
                    f"Selected ids: `{detail.selected_ids}`",
                    f"Missing gold ids: `{detail.missing_gold_ids}`",
                    f"Gold ranks: `{detail.gold_ranks}`",
                    "",
                    "Selected evidence preview:",
                    "",
                    *[f"- {preview}" for preview in detail.selected_evidence_preview],
                    "",
                    "Missing gold evidence preview:",
                    "",
                    *[f"- {preview}" for preview in detail.missing_gold_evidence_preview],
                    "",
                ]
            )

        if include_hits:
            lines.extend(["## Hits", ""])
            hits = [detail for detail in self.details if detail.hit]
            if not hits:
                lines.append("No hits in this run.")
            for detail in hits:
                lines.extend(
                    [
                        f"### HIT: {detail.example_id}",
                        "",
                        f"Question: {detail.question}",
                        "",
                        f"Gold ids: `{detail.gold_ids}`",
                        f"Selected ids: `{detail.selected_ids}`",
                        f"Gold ranks: `{detail.gold_ranks}`",
                        "",
                        "Selected evidence preview:",
                        "",
                        *[f"- {preview}" for preview in detail.selected_evidence_preview],
                        "",
                    ]
                )

        return "\n".join(lines) + "\n"


def evaluate_retrieval_on_examples(
    examples: list[FinQAExample],
    retriever: Retriever,
    config: RetrievalConfig,
    limit: int | None = None,
) -> RetrievalEvaluationSummary:
    """Evaluate whether selected evidence overlaps gold support ids."""
    report = evaluate_retrieval_detailed(
        examples=examples,
        retriever=retriever,
        config=config,
        limit=limit,
    )
    return report.summary


def evaluate_retrieval_detailed(
    examples: list[FinQAExample],
    retriever: Retriever,
    config: RetrievalConfig,
    limit: int | None = None,
    data_path: str | None = None,
    log_path: str | Path | None = None,
    include_hits_in_log: bool = False,
) -> RetrievalEvaluationReport:
    """Evaluate retrieval and optionally write a Markdown log."""
    selected_examples = examples[:limit] if limit is not None else examples
    examples_with_gold = 0
    examples_with_hit = 0
    details: list[RetrievalExampleResult] = []

    for example in selected_examples:
        gold_ids = set(example.gold.supporting_facts)
        if not gold_ids:
            continue

        examples_with_gold += 1
        evidence_units = build_evidence_units(example)
        result = retriever.retrieve(example.runtime.question, evidence_units, config)
        selected_ids = {item.unit.evidence_id for item in result.selected_evidence}
        ranked_by_id = {item.unit.evidence_id: item.rank for item in result.ranked_evidence}
        score_by_id = {item.unit.evidence_id: item.score for item in result.ranked_evidence}
        evidence_by_id = {unit.evidence_id: unit for unit in evidence_units}
        hit = bool(gold_ids & selected_ids)

        if hit:
            examples_with_hit += 1

        details.append(
            RetrievalExampleResult(
                example_id=example.runtime.example_id,
                question=example.runtime.question,
                gold_ids=sorted(gold_ids),
                selected_ids=[item.unit.evidence_id for item in result.selected_evidence],
                hit=hit,
                missing_gold_ids=sorted(gold_ids - selected_ids),
                gold_ranks={gold_id: ranked_by_id.get(gold_id) for gold_id in sorted(gold_ids)},
                selected_evidence_preview=_format_selected_previews(result.selected_evidence, gold_ids),
                missing_gold_evidence_preview=[
                    _format_gold_preview(gold_id, evidence_by_id, score_by_id, ranked_by_id)
                    for gold_id in sorted(gold_ids - selected_ids)
                ],
            )
        )

    recall = examples_with_hit / examples_with_gold if examples_with_gold else 0.0
    summary = RetrievalEvaluationSummary(
        total_examples=len(selected_examples),
        examples_with_gold=examples_with_gold,
        examples_with_hit=examples_with_hit,
        recall_at_selection=recall,
    )
    report = RetrievalEvaluationReport(
        summary=summary,
        details=details,
        config=config,
        data_path=data_path,
    )

    if log_path is not None:
        resolved_log_path = Path(log_path)
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_log_path.write_text(report.to_markdown(include_hits=include_hits_in_log), encoding="utf-8")

    return report


def run_retrieval_evaluation(
    data_path: str | Path,
    config: RetrievalConfig,
    limit: int | None = None,
    log_path: str | Path | None = None,
    include_hits_in_log: bool = False,
) -> RetrievalEvaluationReport:
    """Load a FinQA split, run the configured retriever, and return a report."""
    examples = load_finqa_examples(data_path)
    retriever = build_retriever(config.strategy)
    return evaluate_retrieval_detailed(
        examples=examples,
        retriever=retriever,
        config=config,
        limit=limit,
        data_path=str(data_path),
        log_path=log_path,
        include_hits_in_log=include_hits_in_log,
    )


def _format_selected_previews(selected_evidence, gold_ids: set[str]) -> list[str]:
    sorted_items = sorted(
        selected_evidence,
        key=lambda item: (
            item.unit.evidence_id not in gold_ids,
            item.rank,
            item.unit.evidence_id,
        ),
    )
    return [
        _format_selected_preview(item, is_gold=item.unit.evidence_id in gold_ids)
        for item in sorted_items
    ]


def _format_selected_preview(item, is_gold: bool) -> str:
    marker = "GOLD_MATCH " if is_gold else ""
    return f"{marker}{item.unit.evidence_id} score={item.score:.4f}: {item.unit.text[:180]}"


def _format_gold_preview(
    gold_id: str,
    evidence_by_id: dict,
    score_by_id: dict[str, float],
    rank_by_id: dict[str, int],
) -> str:
    unit = evidence_by_id.get(gold_id)
    if unit is None:
        return f"{gold_id}: <not found in constructed evidence>"
    score = score_by_id.get(gold_id)
    rank = rank_by_id.get(gold_id)
    score_text = f"score={score:.4f}" if score is not None else "score=<missing>"
    rank_text = f"rank={rank}" if rank is not None else "rank=<missing>"
    return f"{gold_id} {score_text} {rank_text}: {unit.text[:240]}"

"""Prompt construction helpers for FinQA reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langchain_core.prompts import PromptTemplate

from src.data.schemas import EvidenceUnit
from src.retrieval.base import RetrievedEvidence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_DIR = PROJECT_ROOT / "configs" / "prompts" / "finqa_prompt_A"


@dataclass(frozen=True, slots=True)
class PromptAssets:
    """Text assets used to assemble the FinQA program prompt."""

    system: str
    evidence_instructions: str
    operation_guide: str
    task_template: str
    few_shot_examples: str = ""


EvidenceLike = EvidenceUnit | RetrievedEvidence


def load_prompt_assets(prompt_dir: str | Path = DEFAULT_PROMPT_DIR) -> PromptAssets:
    """Load prompt sections from a prompt asset directory."""
    prompt_path = Path(prompt_dir)
    return PromptAssets(
        system=_read_prompt_file(prompt_path / "system.txt"),
        evidence_instructions=_read_prompt_file(prompt_path / "evidence_instructions.txt"),
        operation_guide=_read_prompt_file(prompt_path / "operation_guide.txt"),
        few_shot_examples=_read_prompt_file(prompt_path / "few_shot_examples.txt"),
        task_template=_read_prompt_file(prompt_path / "task_template.txt"),
    )


def format_evidence_context(selected_evidence: Sequence[EvidenceLike]) -> str:
    """Render selected evidence as prompt-ready lines.

    The input order is preserved intentionally. Retrieval can decide whether
    that order means global rank, per-source rank, or another selection policy.
    """
    lines = []
    for item in selected_evidence:
        unit = _extract_evidence_unit(item)
        lines.append(f"[{unit.evidence_id}] {unit.text}")
    return "\n".join(lines)


def assemble_reasoning_prompt(
    question: str,
    evidence_context: str,
    assets: PromptAssets | None = None,
) -> str:
    """Assemble the final model-ready prompt from text assets and inputs."""
    prompt_assets = assets or load_prompt_assets()
    return _render_prompt_template(
        template=_build_full_prompt_template(prompt_assets),
        question=question.strip(),
        evidence_context=evidence_context.strip(),
    )


def build_langchain_prompt_template(
    assets: PromptAssets | None = None,
) -> PromptTemplate:
    """Build a LangChain prompt template from the same text assets.

    The plain Python assembler remains the source of truth for prompt wording.
    This adapter lets later LangChain model wrappers use the same contract.
    """
    prompt_assets = assets or load_prompt_assets()
    return PromptTemplate.from_template(_build_full_prompt_template(prompt_assets))


def build_langchain_reasoning_prompt(
    question: str,
    selected_evidence: Sequence[EvidenceLike],
    prompt_dir: str | Path = DEFAULT_PROMPT_DIR,
) -> str:
    """Render the FinQA prompt through the LangChain template adapter."""
    assets = load_prompt_assets(prompt_dir)
    template = build_langchain_prompt_template(assets)
    evidence_context = format_evidence_context(selected_evidence)
    return template.format(
        question=question.strip(),
        evidence_context=evidence_context.strip(),
    )


def build_reasoning_prompt(
    question: str,
    selected_evidence: Sequence[EvidenceLike],
    prompt_dir: str | Path = DEFAULT_PROMPT_DIR,
) -> str:
    """Build the V1 FinQA reasoning prompt from selected evidence."""
    assets = load_prompt_assets(prompt_dir)
    evidence_context = format_evidence_context(selected_evidence)
    return assemble_reasoning_prompt(
        question=question,
        evidence_context=evidence_context,
        assets=assets,
    )


def _read_prompt_file(path: Path) -> str:
    """Read one prompt text file, returning an empty section if omitted."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _extract_evidence_unit(item: EvidenceLike) -> EvidenceUnit:
    """Return the underlying evidence unit from supported prompt inputs."""
    if isinstance(item, RetrievedEvidence):
        return item.unit
    return item


def _build_full_prompt_template(assets: PromptAssets) -> str:
    """Return the complete prompt template in the canonical section order."""
    return "\n\n".join(section.strip() for section in _iter_prompt_sections(assets) if section.strip())


def _iter_prompt_sections(assets: PromptAssets) -> tuple[str, ...]:
    """Return prompt sections in the canonical order used by every adapter."""
    return (
        assets.system,
        assets.evidence_instructions,
        assets.operation_guide,
        assets.few_shot_examples,
        assets.task_template,
    )


def _render_prompt_template(template: str, question: str, evidence_context: str) -> str:
    """Render a prompt template with the same variables as LangChain."""
    return template.format(
        question=question,
        evidence_context=evidence_context,
    )

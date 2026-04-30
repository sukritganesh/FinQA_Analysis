"""Prompt construction helpers for FinQA reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Sequence

import yaml
from langchain_core.prompts import PromptTemplate

from src.data.schemas import EvidenceUnit
from src.retrieval.base import RetrievedEvidence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_DIR = PROJECT_ROOT / "configs" / "prompts" / "finqa_prompt_A"
PROMPT_MANIFEST_FILENAME = "prompt.yaml"
DEFAULT_PROMPT_FILES = (
    "system.txt",
    "evidence_instructions.txt",
    "operation_guide.txt",
    "few_shot_examples.txt",
    "task_template.txt",
)
REQUIRED_TEMPLATE_VARIABLES = frozenset({"question", "evidence_context"})
SUPPORTED_TEMPLATE_VARIABLES = REQUIRED_TEMPLATE_VARIABLES


@dataclass(frozen=True, slots=True)
class PromptSection:
    """One text file included in a prompt template."""

    name: str
    path: Path
    text: str


@dataclass(frozen=True, slots=True)
class PromptAssets:
    """Text assets used to assemble the FinQA program prompt."""

    sections: tuple[PromptSection, ...]

    @property
    def system(self) -> str:
        return self.section_text("system")

    @property
    def evidence_instructions(self) -> str:
        return self.section_text("evidence_instructions")

    @property
    def operation_guide(self) -> str:
        return self.section_text("operation_guide")

    @property
    def few_shot_examples(self) -> str:
        return self.section_text("few_shot_examples")

    @property
    def task_template(self) -> str:
        return self.section_text("task_template")

    def section_text(self, name: str) -> str:
        """Return one section by manifest name or file stem."""
        for section in self.sections:
            if section.name == name or section.path.stem == name:
                return section.text
        return ""


EvidenceLike = EvidenceUnit | RetrievedEvidence


def load_prompt_assets(prompt_dir: str | Path = DEFAULT_PROMPT_DIR) -> PromptAssets:
    """Load prompt sections from a prompt asset directory."""
    prompt_path = Path(prompt_dir)
    sections = _load_prompt_sections(prompt_path)
    assets = PromptAssets(sections=sections)
    _validate_prompt_template(_build_full_prompt_template(assets), prompt_path)
    return assets


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


def _load_prompt_sections(prompt_path: Path) -> tuple[PromptSection, ...]:
    manifest_path = prompt_path / PROMPT_MANIFEST_FILENAME
    if manifest_path.exists():
        return _load_manifest_prompt_sections(prompt_path, manifest_path)
    return _load_default_prompt_sections(prompt_path)


def _load_default_prompt_sections(prompt_path: Path) -> tuple[PromptSection, ...]:
    sections = []
    for filename in DEFAULT_PROMPT_FILES:
        path = prompt_path / filename
        text = _read_prompt_file(path)
        if text:
            sections.append(PromptSection(name=path.stem, path=path, text=text))
    return tuple(sections)


def _load_manifest_prompt_sections(
    prompt_path: Path,
    manifest_path: Path,
) -> tuple[PromptSection, ...]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt manifest must be a YAML mapping: {manifest_path}")

    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ValueError(
            f"Prompt manifest must define a non-empty 'sections' list: {manifest_path}"
        )

    sections = []
    for index, raw_section in enumerate(raw_sections):
        name, relative_path = _parse_manifest_section(raw_section, index, manifest_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Prompt manifest section paths must stay inside the prompt folder: {manifest_path}"
            )

        path = prompt_path / relative_path
        if not path.exists():
            raise ValueError(f"Prompt manifest references missing file: {path}")

        text = path.read_text(encoding="utf-8").strip()
        if text:
            sections.append(PromptSection(name=name, path=path, text=text))

    return tuple(sections)


def _parse_manifest_section(
    raw_section: object,
    index: int,
    manifest_path: Path,
) -> tuple[str, Path]:
    if isinstance(raw_section, str):
        path = Path(raw_section)
        return path.stem, path

    if not isinstance(raw_section, dict):
        raise ValueError(
            f"Prompt manifest section #{index + 1} must be a filename or mapping: {manifest_path}"
        )

    raw_file = raw_section.get("file") or raw_section.get("path")
    if not isinstance(raw_file, str) or not raw_file.strip():
        raise ValueError(
            f"Prompt manifest section #{index + 1} must include a non-empty file: {manifest_path}"
        )

    path = Path(raw_file)
    raw_name = raw_section.get("name", path.stem)
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError(
            f"Prompt manifest section #{index + 1} must include a valid name: {manifest_path}"
        )

    return raw_name.strip(), path


def _extract_evidence_unit(item: EvidenceLike) -> EvidenceUnit:
    """Return the underlying evidence unit from supported prompt inputs."""
    if isinstance(item, RetrievedEvidence):
        return item.unit
    return item


def _build_full_prompt_template(assets: PromptAssets) -> str:
    """Return the complete prompt template in the canonical section order."""
    return "\n\n".join(section.text.strip() for section in assets.sections if section.text.strip())


def _validate_prompt_template(template: str, prompt_path: Path) -> None:
    variables = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None and field_name != ""
    }
    missing = REQUIRED_TEMPLATE_VARIABLES - variables
    if missing:
        missing_list = ", ".join(sorted(f"{{{name}}}" for name in missing))
        raise ValueError(
            f"Prompt template in {prompt_path} is missing required variable(s): {missing_list}"
        )

    unsupported = variables - SUPPORTED_TEMPLATE_VARIABLES
    if unsupported:
        unsupported_list = ", ".join(sorted(f"{{{name}}}" for name in unsupported))
        raise ValueError(
            f"Prompt template in {prompt_path} contains unsupported variable(s): {unsupported_list}"
        )


def _render_prompt_template(template: str, question: str, evidence_context: str) -> str:
    """Render a prompt template with the same variables as LangChain."""
    return template.format(
        question=question,
        evidence_context=evidence_context,
    )

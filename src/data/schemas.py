"""Core typed data structures for normalized FinQA examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RuntimeInputs:
    """Fields that are safe to use during inference."""

    example_id: str
    filename: str | None
    question: str
    pre_text: list[str]
    post_text: list[str]
    table: list[list[str]]


@dataclass(slots=True)
class GoldTargets:
    """Gold annotations used only for evaluation and analysis.

    These fields are optional because private-test style examples do not include
    references. Runtime code should be valid when this object is empty.
    """

    answer: str | None = None
    executable_answer: str | None = None
    program: str | None = None
    program_nested: str | None = None
    supporting_facts: dict[str, str] = field(default_factory=dict)
    ann_text_rows: list[int] = field(default_factory=list)
    ann_table_rows: list[int] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    explanation: str | None = None

    @property
    def has_labels(self) -> bool:
        """Return whether this example includes any gold supervision."""
        return any(
            [
                self.answer is not None,
                self.executable_answer is not None,
                self.program is not None,
                bool(self.supporting_facts),
            ]
        )


@dataclass(slots=True)
class ExampleMetadata:
    """Optional source artifacts useful for inspection and debugging."""

    table_original: Any | None = None
    model_input: Any | None = None
    tfidf_topn: Any | None = None
    text_retrieved: Any | None = None
    text_retrieved_all: Any | None = None
    table_retrieved: Any | None = None
    table_retrieved_all: Any | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceUnit:
    """A searchable evidence fragment derived from text or table content."""

    evidence_id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FinQAExample:
    """Normalized example with explicit runtime/gold separation."""

    runtime: RuntimeInputs
    gold: GoldTargets = field(default_factory=GoldTargets)
    metadata: ExampleMetadata = field(default_factory=ExampleMetadata)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataclass to a plain Python dictionary."""
        return asdict(self)

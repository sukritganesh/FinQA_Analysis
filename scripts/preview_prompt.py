"""Preview the exact FinQA prompt produced for one retrieved example.

Useful commands:

    # Preview prompt with balanced 2 text + 2 table retrieval.
    .venv/bin/python scripts/preview_prompt.py --input data/raw/test.json --index 0 --mode by_source --top-k-text 2 --top-k-table 2

    # Preview prompt with flexible top 4 overall retrieval.
    .venv/bin/python scripts/preview_prompt.py --input data/raw/test.json --index 0 --mode combined --top-k 4

    # Write the prompt and metadata to files.
    .venv/bin/python scripts/preview_prompt.py --input data/raw/test.json --index 0 --mode by_source --top-k-text 2 --top-k-table 2 --output reports/prompts/test_example_0_prompt.txt --metadata-output reports/prompts/test_example_0_metadata.md

    # Print only the prompt text that will be sent to the model.
    .venv/bin/python scripts/preview_prompt.py --input data/raw/test.json --index 0 --mode combined --top-k 4 --prompt-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import shorten

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.evidence import build_evidence_units
from src.data.loader import load_finqa_examples
from src.data.schemas import FinQAExample
from src.llm.prompts import build_reasoning_prompt
from src.retrieval.base import RetrievalConfig
from src.retrieval.factory import build_retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Path to a FinQA JSON split.")
    parser.add_argument("--index", type=int, default=0, help="Zero-based example index to preview.")
    parser.add_argument("--example-id", default=None, help="Optional FinQA example ID. Overrides --index when provided.")
    parser.add_argument("--strategy", default="bm25", choices=["bm25"], help="Retrieval strategy.")
    parser.add_argument("--mode", default="by_source", choices=["by_source", "combined"], help="Retrieval mode.")
    parser.add_argument("--top-k", type=int, default=4, help="Combined-mode top-k.")
    parser.add_argument("--top-k-text", type=int, default=2, help="By-source text top-k.")
    parser.add_argument("--top-k-table", type=int, default=2, help="By-source table top-k.")
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=None,
        help="Optional prompt template directory. Defaults to src.llm.prompts.DEFAULT_PROMPT_DIR.",
    )
    parser.add_argument("--prompt-only", action="store_true", help="Print only the final prompt text.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the final prompt text.")
    parser.add_argument("--metadata-output", type=Path, default=None, help="Optional path to write prompt metadata.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = load_finqa_examples(args.input)
    example = _select_example(examples, index=args.index, example_id=args.example_id)

    evidence_units = build_evidence_units(example)
    config = RetrievalConfig(
        strategy=args.strategy,
        mode=args.mode,
        top_k=args.top_k,
        top_k_text=args.top_k_text,
        top_k_table=args.top_k_table,
    )
    retriever = build_retriever(config.strategy)
    retrieval = retriever.retrieve(
        question=example.runtime.question,
        evidence_units=evidence_units,
        config=config,
    )
    prompt_kwargs = {}
    if args.prompt_dir is not None:
        prompt_kwargs["prompt_dir"] = args.prompt_dir
    prompt = build_reasoning_prompt(
        question=example.runtime.question,
        selected_evidence=retrieval.selected_evidence,
        **prompt_kwargs,
    )
    metadata = _build_prompt_metadata(
        path=args.input,
        example=example,
        config=config,
        prompt=prompt,
        selected_evidence=retrieval.selected_evidence,
    )

    if args.output is not None:
        _write_text(args.output, prompt)
    if args.metadata_output is not None:
        _write_text(args.metadata_output, _format_metadata_markdown(metadata))

    if args.prompt_only:
        print(prompt)
        _print_write_summary(args.output, args.metadata_output)
        return

    print(_build_preview_header(args.input, example, config))
    print("\nPrompt metadata:")
    print(_format_metadata_console(metadata))
    print("\nSelected evidence:")
    for item in retrieval.selected_evidence:
        preview = shorten(item.unit.text, width=130, placeholder="...")
        source_rank = "-" if item.source_rank is None else str(item.source_rank)
        print(
            f"- rank={item.rank} source_rank={source_rank} "
            f"score={item.score:.4f} id={item.unit.evidence_id} source={item.unit.source}: {preview}"
        )

    print("\n--- PROMPT START ---")
    print(prompt)
    print("--- PROMPT END ---")
    _print_write_summary(args.output, args.metadata_output)


def _select_example(examples: list[FinQAExample], index: int, example_id: str | None) -> FinQAExample:
    if example_id is not None:
        for example in examples:
            if example.runtime.example_id == example_id:
                return example
        msg = f"Example ID not found: {example_id}"
        raise ValueError(msg)

    if index < 0 or index >= len(examples):
        msg = f"Example index {index} is outside the split range 0..{len(examples) - 1}"
        raise IndexError(msg)
    return examples[index]


def _build_preview_header(path: Path, example: FinQAExample, config: RetrievalConfig) -> str:
    if config.mode == "combined":
        retrieval_summary = f"strategy={config.strategy}, mode={config.mode}, top_k={config.top_k}"
    else:
        retrieval_summary = (
            f"strategy={config.strategy}, mode={config.mode}, "
            f"top_k_text={config.top_k_text}, top_k_table={config.top_k_table}"
        )

    return "\n".join(
        [
            "Prompt preview",
            f"Input: {path}",
            f"Example ID: {example.runtime.example_id}",
            f"Question: {example.runtime.question}",
            f"Retrieval: {retrieval_summary}",
        ]
    )


def _build_prompt_metadata(
    path: Path,
    example: FinQAExample,
    config: RetrievalConfig,
    prompt: str,
    selected_evidence: list,
) -> dict[str, object]:
    evidence_counts = _count_selected_sources(selected_evidence)
    prompt_characters = len(prompt)
    prompt_lines = len(prompt.splitlines())
    return {
        "input": str(path),
        "example_id": example.runtime.example_id,
        "question": example.runtime.question,
        "retrieval_strategy": config.strategy,
        "retrieval_mode": config.mode,
        "top_k": config.top_k,
        "top_k_text": config.top_k_text,
        "top_k_table": config.top_k_table,
        "selected_evidence_count": len(selected_evidence),
        "selected_text_count": evidence_counts.get("text", 0),
        "selected_table_count": evidence_counts.get("table", 0),
        "prompt_characters": prompt_characters,
        "prompt_lines": prompt_lines,
        "estimated_tokens_4_5_chars": round(prompt_characters / 4.5),
    }


def _count_selected_sources(selected_evidence: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in selected_evidence:
        source = str(item.unit.source)
        counts[source] = counts.get(source, 0) + 1
    return counts


def _format_metadata_console(metadata: dict[str, object]) -> str:
    return "\n".join(
        [
            f"- prompt_characters: {metadata['prompt_characters']}",
            f"- prompt_lines: {metadata['prompt_lines']}",
            f"- estimated_tokens_4_5_chars: {metadata['estimated_tokens_4_5_chars']}",
            f"- selected_evidence_count: {metadata['selected_evidence_count']}",
            f"- selected_text_count: {metadata['selected_text_count']}",
            f"- selected_table_count: {metadata['selected_table_count']}",
        ]
    )


def _format_metadata_markdown(metadata: dict[str, object]) -> str:
    lines = [
        "# Prompt Preview Metadata",
        "",
        f"- Input: `{metadata['input']}`",
        f"- Example ID: `{metadata['example_id']}`",
        f"- Question: {metadata['question']}",
        f"- Retrieval strategy: `{metadata['retrieval_strategy']}`",
        f"- Retrieval mode: `{metadata['retrieval_mode']}`",
        f"- Combined top-k: `{metadata['top_k']}`",
        f"- By-source text top-k: `{metadata['top_k_text']}`",
        f"- By-source table top-k: `{metadata['top_k_table']}`",
        f"- Selected evidence count: `{metadata['selected_evidence_count']}`",
        f"- Selected text count: `{metadata['selected_text_count']}`",
        f"- Selected table count: `{metadata['selected_table_count']}`",
        f"- Prompt characters: `{metadata['prompt_characters']}`",
        f"- Prompt lines: `{metadata['prompt_lines']}`",
        f"- Estimated tokens at 4.5 chars/token: `{metadata['estimated_tokens_4_5_chars']}`",
        "",
    ]
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _print_write_summary(output: Path | None, metadata_output: Path | None) -> None:
    if output is not None:
        print(f"\nWrote prompt to: {output}")
    if metadata_output is not None:
        print(f"Wrote metadata to: {metadata_output}")


if __name__ == "__main__":
    main()

"""Inspect one or more FinQA examples from a local JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_finqa_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to a FinQA JSON file.")
    parser.add_argument("--limit", type=int, default=1, help="Number of examples to print.")
    parser.add_argument("--summary", action="store_true", help="Print split-level summary before examples.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = load_finqa_examples(args.input)

    if args.summary:
        labeled = sum(example.gold.has_labels for example in examples)
        print(
            json.dumps(
                {
                    "path": str(args.input),
                    "examples": len(examples),
                    "labeled_examples": labeled,
                    "unlabeled_examples": len(examples) - labeled,
                },
                indent=2,
            )
        )

    for example in examples[: args.limit]:
        print(json.dumps(example.to_dict(), indent=2))


if __name__ == "__main__":
    main()

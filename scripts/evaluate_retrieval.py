"""Run retrieval evaluation over a local FinQA split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.retrieval import run_retrieval_evaluation
from src.retrieval.base import RetrievalConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Path to a FinQA JSON split.")
    parser.add_argument("--limit", type=int, default=None, help="Number of examples to evaluate. Omit for the full split.")
    parser.add_argument("--strategy", default="bm25", choices=["bm25"], help="Retrieval strategy.")
    parser.add_argument("--mode", default="by_source", choices=["by_source", "combined"], help="Retrieval mode.")
    parser.add_argument("--top-k", type=int, default=6, help="Combined-mode top-k.")
    parser.add_argument("--top-k-text", type=int, default=3, help="By-source text top-k.")
    parser.add_argument("--top-k-table", type=int, default=3, help="By-source table top-k.")
    parser.add_argument("--log-path", type=Path, default=None, help="Optional Markdown log output path.")
    parser.add_argument("--include-hits", action="store_true", help="Include successful examples in the log.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RetrievalConfig(
        strategy=args.strategy,
        mode=args.mode,
        top_k=args.top_k,
        top_k_text=args.top_k_text,
        top_k_table=args.top_k_table,
    )
    report = run_retrieval_evaluation(
        data_path=args.input,
        config=config,
        limit=args.limit,
        log_path=args.log_path,
        include_hits_in_log=args.include_hits,
    )

    print(report.to_markdown(include_hits=args.include_hits))


if __name__ == "__main__":
    main()

"""Run a small local demo using the starter pipeline skeleton."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.cli import run_demo_question


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Question to send through the demo pipeline.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_demo_question(args.question)
    print(result)


if __name__ == "__main__":
    main()

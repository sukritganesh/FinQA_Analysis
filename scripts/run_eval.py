"""Run the starter evaluation skeleton on prediction and gold files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.runner import load_prediction_map, run_answer_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions JSON.")
    parser.add_argument("--gold", type=Path, required=True, help="Path to FinQA gold JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictions = load_prediction_map(args.predictions)
    result = run_answer_evaluation(gold_path=args.gold, predictions=predictions)
    print(result.to_pretty_json())


if __name__ == "__main__":
    main()

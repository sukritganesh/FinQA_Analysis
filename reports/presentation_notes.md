# Presentation Notes

## Opening

- Explain the task as financial QA over mixed text and table evidence.
- Emphasize explainability and numerical correctness.

## Architecture

- Show the staged pipeline.
- Highlight runtime vs. gold-label separation.
- Explain why retrieval comes before generation.

## Implementation Choices

- Small, modular Python packages
- Constrained interfaces
- Deterministic arithmetic utilities

## Early Risks

- Table rendering quality affects retrieval.
- Output schema design affects parser reliability.
- Numeric normalization matters for evaluation.

## Demo Plan

- Show one example through the pipeline.
- Display retrieved evidence and final parsed output.

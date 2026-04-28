# 10. Report And Presentation

## Purpose

Prepare the project story for a take-home submission. The report should explain not only what was built, but why the architecture is appropriate for FinQA.

## Core Story

FinQA is not ordinary open-ended QA. It requires:

- selecting the right evidence from text and tables
- preserving source ids for debugging
- prompting the model with constrained context
- forcing parseable model outputs
- executing arithmetic deterministically
- evaluating numeric correctness with tolerance

The project is designed as a staged pipeline so each of those responsibilities can be inspected independently.

## Suggested Report Structure

1. Problem overview
2. Dataset structure and runtime/gold separation
3. Evidence construction
4. BM25 retrieval baseline
5. Prompt and output contract
6. Model invocation with vLLM and optional OpenAI provider
7. Deterministic parser and executor
8. Evaluation methodology
9. Results and error analysis
10. Limitations and next steps

## Points To Emphasize

- The system is inference-first; no fine-tuning is required for the first implementation.
- Runtime inputs are separated from gold evaluation targets to avoid label leakage.
- Retrieval is classical and inspectable before adding heavier methods.
- The model is not treated as a generic chatbot; it is constrained to return a direct answer or executable program.
- Python performs arithmetic instead of trusting free-form model math.
- Evaluation distinguishes missing answers, parse failures, execution failures, retrieval misses, and wrong final answers.

## Useful Artifacts

- `reports/presentation_notes.md`
- `reports/test_results/`
- `reports/prompts/`
- `configs/runs/`
- `docs/stages/`

## Suggested Result Tables

For the final writeup, include compact tables for:

- retrieval configuration and support-hit behavior
- model and prompt variant used
- final-answer accuracy
- answered vs unanswered examples
- parse/execution failure counts
- a few representative success and failure examples

## Limitations To Discuss

- BM25 can miss evidence when wording differs from the question.
- Prompt following varies by model size and provider.
- The strict parser improves reliability but rejects useful answers that are not in contract.
- Table aggregation uses simple row matching.
- vLLM setup is environment-specific, especially on Windows-native workflows.
- Results depend on local model choice, quantization, and prompt variant.

## Possible Next Steps

- Add retrieval boosts for years, metrics, and comparison terms.
- Try hybrid retrieval or reranking after the BM25 baseline is documented.
- Add richer error-analysis reports.
- Compare prompt variants on the same sample.
- Evaluate additional local models.
- Consider fine-tuning only after the inference baseline is well understood.

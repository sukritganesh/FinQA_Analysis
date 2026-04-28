# 04. Prompt Generation

## Purpose

Turn a question and selected evidence into a complete model prompt. The prompt constrains the model to produce a one-line answer that the parser can handle.

## What This Stage Does

- Formats selected evidence with ids.
- Loads prompt text from configurable prompt directories.
- Assembles the final prompt string.
- Exposes a thin LangChain `PromptTemplate` adapter for prompt rendering.
- Wraps prompt assembly in a LangGraph node.

## Key Files

- `src/llm/prompts.py`
- `src/graph/prompting.py`
- `configs/prompts/finqa_prompt_A/`
- `configs/prompts/finqa_prompt_B_compact/`
- `tests/test_prompt_construction.py`
- `tests/test_prompt_generation_graph.py`

## Prompt Contract

The prompt should describe the task and the provided evidence.

It should also reference the executable output contract and provide examples of good and bad outputs.

The shared output format is documented in [Executable Program Contract](../details/executable_programs.md).

Finally, the prompt can provide examples with an input question, evidence, output program, and answer for few-shot prompting.

The actual question and evidence will be at the end.

Sections can be added or removed.

## Prompt Assets

Prompt wording lives outside Python source code. A prompt directory can contain files such as:

- `system.txt`
- `evidence_instructions.txt`
- `operation_guide.txt`
- `few_shot_examples.txt`
- `task_template.txt`

Two existing prompt variants are provided. `finqa_prompt_A` is fuller. `finqa_prompt_B_compact` is shorter and useful for smaller context windows.

## LangChain Role

LangChain is used lightly for prompt templates and variable management. Plain Python prompt assembly remains easy to inspect and test.

## Graph Integration

The prompt node expects a question or selected example plus retrieved evidence:

```text
selected_example/question + retrieved_evidence + prompt_dir -> prompt
```

It does not call the model, parse output, or execute arithmetic.

## Verification

Tests cover:

- prompt asset loading
- evidence-context formatting
- selected evidence order
- question insertion
- LangChain rendering parity with the plain Python builder
- LangGraph prompt generation

## Possible Improvements

- Keep a short prompt variant for cheaper/lower-context models.
- Add small prompt examples only when they improve parse success.
- Record prompt variants with run configs so evaluation results are reproducible.

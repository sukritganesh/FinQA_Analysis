# FinQA Prompt Templates

This folder stores prompt text for the V1 reasoning/generation step.
Prompt folders are intentionally template-specific so we can compare variants.

Current variants:

- `finqa_prompt_A`: fuller instruction set with six few-shot examples.
- `finqa_prompt_B_compact`: shorter instruction set with three compact examples, designed to reduce prompt load and test whether less instruction text improves local model focus.

The prompt is split into small files so we can edit instructions without digging through Python code.

Suggested assembly order:

1. `system.txt`
2. `evidence_instructions.txt`
3. `operation_guide.txt`
4. `few_shot_examples.txt`
5. `task_template.txt`

Prompt text files are treated as optional by the Python loader. Missing files are skipped, which makes it easier to experiment with smaller prompt variants.

The Python prompt builder should inject:

- `{question}`
- `{evidence_context}`

The expected model output is exactly one line:

- a FinQA-style executable program
- or a direct answer value when no computation is needed

Preview a real assembled prompt with:

```bash
.venv/bin/python scripts/preview_prompt.py --input data/raw/test.json --index 0 --mode by_source --top-k-text 2 --top-k-table 2 --prompt-dir configs/prompts/finqa_prompt_A
```

Write the same preview prompt and metadata to files with:

```bash
.venv/bin/python scripts/preview_prompt.py --input data/raw/test.json --index 0 --mode by_source --top-k-text 2 --top-k-table 2 --prompt-dir configs/prompts/finqa_prompt_A --output reports/prompts/test_example_0_prompt.txt --metadata-output reports/prompts/test_example_0_metadata.md
```

Note: You are free to construct your own prompt files!
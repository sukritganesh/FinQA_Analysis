# FinQA Prompt Templates

This folder stores prompt text for the V1 reasoning/generation step.
Prompt folders are intentionally template-specific so we can compare variants.

Current variants:

- `finqa_prompt_A`: fuller instruction set with six few-shot examples.
- `finqa_prompt_B_compact`: shorter instruction set with three compact examples, designed to reduce prompt load and test whether less instruction text improves local model focus.

The prompt is split into small files so we can edit instructions without digging through Python code.
Each prompt folder can include a `prompt.yaml` manifest that controls which text files are composed and in what order.

Default assembly order when no manifest is present:

1. `system.txt`
2. `evidence_instructions.txt`
3. `operation_guide.txt`
4. `few_shot_examples.txt`
5. `task_template.txt`

Example manifest:

```yaml
sections:
  - system.txt
  - evidence_instructions.txt
  - operation_guide.txt
  - few_shot_examples.txt
  - task_template.txt
```

You can use any filenames and any number of files:

```yaml
sections:
  - file: intro.txt
    name: intro
  - file: examples/short_examples.txt
    name: examples
  - task.txt
```

If `prompt.yaml` is present, every listed file must exist. If no manifest is present, the default files are treated as optional and missing files are skipped.

The Python prompt builder should inject:

- `{question}`
- `{evidence_context}`

Those two placeholders must appear somewhere in the composed prompt.

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

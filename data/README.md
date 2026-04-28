# Data Directory

This folder holds local FinQA data files used by the pipeline.

## Expected Layout

```text
data/
  raw/
    train.json
    dev.json
    test.json
    private_test.json
  processed/
    <add-custom-data-here>
```

`data/raw/` should contain the original FinQA split files. `data/processed/` can contain small sampled or transformed files used for local experiments.

## Notes

- Keep raw dataset files unchanged when possible.
- Put derived samples or temporary experiment inputs under `data/processed/`.
- The code uses `pathlib`, so paths should work from Git Bash, PowerShell, or Linux shells when commands are run from the repository root.
- Dataset structure and field meanings are documented in [FinQA Dataset](../docs/details/finqa_dataset.md).

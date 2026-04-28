# 07. Validation And Formatting

## Purpose

Package pipeline outputs into a stable `Prediction` object that evaluation and reporting code can consume.

This stage normalizes and preserves artifacts. It does not decide whether an answer is correct.

## What This Stage Does

- Builds one prediction record for one FinQA example.
- Preserves the raw final answer.
- Stores a normalized answer for comparison.
- Keeps model output, parsed output, execution result, and errors available for analysis.
- Handles successful, partial, and failed predictions consistently.

## Key Files

- `src/eval/prediction.py`
- `src/graph/validation.py`
- `tests/test_prediction_formatting.py`

## Prediction Shape

`Prediction` contains:

- `example_id`
- `question`
- `answer`
- `normalized_answer`
- `model_output_text`
- `parsed_output`
- `execution_result`
- `errors`

This gives evaluation enough context to compare final answers while still exposing parser, executor, and model-call failures.

## Normalization

Normalization is intentionally light:

- preserve the raw final answer as `answer`
- store a comparison-ready form as `normalized_answer`
- lowercase yes/no answers
- trim and standardize answer text
- leave correctness decisions to evaluation

## Graph Integration

The validation node runs after deterministic execution:

```text
final_answer + parsed_output + execution_result + model_output_text -> prediction
```

It packages artifacts even when errors exist, because failed examples are still important for debugging and evaluation summaries.

## Verification

Tests cover:

- successful predictions
- missing final answers
- propagated errors
- normalized answer fields

## Notes

This stage is deliberately small. More complex scoring, numeric tolerance, and gold comparisons belong in the evaluation stage.

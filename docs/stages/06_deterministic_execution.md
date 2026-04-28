# 06. Deterministic Execution

## Purpose

Convert the model's one-line output into a final answer using deterministic Python whenever possible.

The model identifies values and operations. Python parses, validates, and executes the math.

## What This Stage Does

- Parses direct answers and linear operation traces.
- Rejects malformed or unsupported outputs.
- Executes arithmetic with `Decimal`.
- Resolves references such as `#0` and `#1`.
- Supports simple table-row aggregations.
- Records parse and execution errors separately.

The executable output format is documented in [Executable Program Contract](../details/executable_programs.md).

## Key Files

- `src/llm/parser.py`
- `src/tools/calculator.py`
- `src/tools/executor.py`
- `src/graph/execution.py`
- `tests/test_parser.py`
- `tests/test_calculator.py`
- `tests/test_executor.py`
- `tests/test_execution_graph.py`

## Accepted Output Forms

The strict parser accepts the direct-answer and linear-program forms defined in [Executable Program Contract](../details/executable_programs.md).

It rejects prose, JSON, labels, nested expressions, unsupported operations, and invalid references.

## Supported Operations

Supported operations are listed in [Executable Program Contract](../details/executable_programs.md). The parser and executor should stay aligned with that document and with the prompt assets.

The parser also accepts FinQA gold constants such as `const_100` and `const_m1` so dataset programs can be tested, even though prompts ask models to emit ordinary numbers.

## Execution Behavior

For arithmetic programs, each step executes in order:

```text
subtract(153.7, 139.9), divide(#0, 139.9)
```

Execution trace:

```text
#0 = 13.8
#1 = 0.098641887...
final = #1
```

For direct answers, execution normalizes and returns the value without using the calculator.

## Table Operations

Table operations are reserved for aggregating numeric values across one table row. The executor reads the original table from the selected example, finds the row by normalized row name, extracts numeric cells, and applies the aggregation.

Single-cell lookups should usually be copied from retrieved evidence into ordinary arithmetic instead of using `table_*`. See [Executable Program Contract](../details/executable_programs.md) for examples.

## Error Categories

Parse errors include:

- unsupported operation
- malformed function call
- nested expression
- invalid or future reference
- prose/non-one-line output

Execution errors include:

- division by zero
- non-numeric arguments
- missing previous result
- missing table
- table row not found
- table row with no numeric values

## LangGraph Integration

Parsing and execution are separate graph nodes:

```text
model_output_text -> parse_model_output -> parsed_output
parsed_output + selected_example.table -> execute_parsed_output -> final_answer
```

This makes it easy to tell whether a failure came from model formatting, parser rules, or deterministic execution.

## Verification

Tests cover:

- direct numeric and yes/no answers
- arithmetic programs
- chained references
- FinQA constants
- malformed outputs
- table aggregation
- graph parse/execution behavior
- supported real FinQA gold programs

## Possible Improvements

- Add conservative fuzzy matching for table row names if exact normalized matching misses useful cases.
- Add more operation types only when supported by prompts, parser, executor, and tests together.

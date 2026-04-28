# Executable Program Contract

This document defines the one-line output format shared by prompt generation, parsing, deterministic execution, and evaluation debugging.

The model should return either:

- a direct answer
- or a linear executable program

It should not return JSON, Markdown, labels, evidence ids, or explanatory prose.

## Direct Answers

Use a direct answer when no arithmetic is needed:

```text
yes
```

```text
no
```

```text
94
```

```text
3.8
```

Direct answers are normalized during deterministic execution and answer evaluation.

## Linear Programs

Use a program when arithmetic or comparison is needed:

```text
subtract(5829, 5735)
```

Multiple steps are separated by commas. Later steps can refer to earlier results with `#0`, `#1`, and so on:

```text
subtract(153.7, 139.9), divide(#0, 139.9)
```

This means:

```text
#0 = subtract(153.7, 139.9)
answer = divide(#0, 139.9)
```

Nested expressions are intentionally not part of the contract. Use linear references instead.

Avoid:

```text
divide(subtract(153.7, 139.9), 139.9)
```

Prefer:

```text
subtract(153.7, 139.9), divide(#0, 139.9)
```

## Supported Operations

Arithmetic:

- `add(a, b)`
- `subtract(a, b)`
- `multiply(a, b)`
- `divide(a, b)`

Comparison:

- `greater(a, b)`

Table-row aggregation:

- `table_sum(name, none)`
- `table_average(name, none)`
- `table_max(name, none)`
- `table_min(name, none)`

`greater(a, b)` returns `yes` when `a > b`; otherwise it returns `no`.

## Numbers

Programs should use ordinary numeric literals:

```text
subtract(193.5, 100), divide(#0, 100)
```

```text
multiply(607, 18.13), multiply(#0, 1000)
```

The parser also accepts FinQA annotation constants such as `const_100` and `const_m1` so gold programs can be tested, but prompts should ask models to emit ordinary numbers.

Financial formatting such as `$1,234` and `(12)` is normalized during execution. Percent signs are not accepted in model programs; the model should output decimal or ordinary numeric values instead.

## Table Operations

Table operations are only for aggregating numeric values across one table row.

Example:

```text
table_min(expected volatility, none), table_max(expected volatility, none), subtract(#1, #0)
```

The executor:

- reads the original table from the selected FinQA example
- finds the row by normalized row name
- extracts numeric values from the row
- applies the aggregation

Do not use table operations to retrieve a single cell value when retrieved evidence already contains that value.

Example evidence:

```text
[table_8] the 2015 net revenue of amount ( in millions ) is $ 5829 ;
[table_1] the 2014 net revenue of amount ( in millions ) is $ 5735 ;
```

Preferred output:

```text
subtract(5829, 5735)
```

Avoid:

```text
subtract(table_sum(2015 net revenue, none), table_sum(2014 net revenue, none))
```

## Invalid Outputs

These outputs are rejected by the strict parser:

```text
The answer is subtract(5829, 5735).
```

```json
{"program": "subtract(5829, 5735)"}
```

```text
Program: subtract(5829, 5735)
```

```text
subtract(10, 4). The answer is 6.
```

## Why This Format

The format is intentionally small:

- prompts can explain it clearly
- the parser can reject malformed outputs reliably
- Python can execute arithmetic deterministically
- failures are easy to categorize as formatting, parsing, execution, or reasoning errors

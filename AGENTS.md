# AGENTS.md

## Project Overview

This repository is for a take-home assignment based on the **FinQA** dataset.

The goal is to build a practical, modular, explainable question-answering system for **financial numerical reasoning** over mixed evidence from:

- financial text before a table
- a financial table
- financial text after a table

The system should answer questions by:

1. loading and normalizing FinQA examples
2. converting document content into searchable evidence units
3. retrieving relevant evidence for a question
4. passing that evidence into an LLM reasoning/generation step
5. optionally executing arithmetic deterministically in Python
6. producing a final answer
7. evaluating predictions against FinQA gold annotations

This is an **inference-first** project. The first version should focus on:
- retrieval
- prompting
- structured outputs
- deterministic numeric execution
- evaluation

Do **not** assume that model fine-tuning is part of the first implementation unless explicitly requested later.

---

## Core Design Philosophy

This project should be:

- modular
- readable
- easy to debug
- easy to explain in a report or presentation
- realistic, but not overengineered

Prefer:
- simple Python
- explicit function names
- small modules
- typed data structures where useful
- docstrings on important public functions
- TODO comments where implementation is intentionally deferred

Avoid:
- giant abstractions
- speculative enterprise architecture
- unnecessary design patterns
- complicated autonomous agent loops
- heavy framework code unless it serves a clear purpose

This is a take-home assignment, not a production microservices platform.

---

## What the System Should Do

The intended pipeline is a staged workflow for FinQA.

High-level stages:

1. **Data loading**
   - load FinQA JSON examples
   - normalize fields into internal structures

2. **Evidence construction**
   - convert `pre_text` and `post_text` into sentence-level evidence units
   - convert table rows into text-like evidence units suitable for retrieval and prompting

3. **Retrieval**
   - score evidence units for relevance to the question
   - return top-k evidence items

4. **Reasoning / generation**
   - call a local LLM
   - prompt the model with retrieved evidence
   - ask for either a direct answer or a structured reasoning output

5. **Deterministic execution**
   - when possible, execute arithmetic in Python instead of trusting free-form natural-language math

6. **Validation / formatting**
   - normalize and validate outputs
   - prepare a clean final answer

7. **Evaluation**
   - compare predicted outputs to gold answer
   - optionally compare predicted reasoning/program to gold program
   - optionally compare retrieved evidence to gold support facts

---

## Tooling Expectations

This project is expected to make meaningful use of:

- **vLLM** for local model serving / inference
- **LangChain** for prompts, model interfaces, parsing, and structured outputs
- **LangGraph** for high-level workflow orchestration

Interpret the roles like this:

### vLLM
Use as the local inference / serving layer for a Hugging Face model.
The application should be able to call the model through a clean interface.

### LangChain
Use for:
- prompt templates
- model invocation wrappers
- structured output parsing
- formatting retrieved evidence for model input
- optional calculator/tool integration

### LangGraph
Use for:
- workflow state
- graph nodes and edges
- orchestration of retrieval → reasoning → execution → validation

Do not force these tools into places where plain Python would be clearer.

---

## Current Scope Assumptions

Assume the following for the first implementation:

- the project uses **FinQA JSON data**
- the project runs locally
- the first iteration is **inference only**
- no fine-tuning is required for the first iteration
- a local Hugging Face model will be served via vLLM
- retrieval will operate over evidence units derived from sentences and table rows
- evaluation should include final-answer comparison at minimum

Do not add:
- web UI
- distributed infrastructure
- database-backed production systems
- authentication
- cloud deployment complexity
- autonomous agents beyond the assignment needs

---

## Internal Data Representation

The code should clearly separate:

### Runtime inputs
Data available to the system at inference time, such as:
- example ID
- question
- text evidence
- table evidence
- metadata

### Gold evaluation targets
Data only for evaluation / analysis, such as:
- gold answer
- gold reasoning program
- gold supporting evidence

Do not design code that accidentally relies on gold fields during inference.

A normalized internal example structure should make this separation obvious.

---

## Retrieval Expectations

The retriever should work over candidate evidence units built from:

- individual text sentences
- table rows rendered into text-like form

Retrieval should be modular and inspectable.

The code should support:
- retrieving top-k evidence units
- inspecting retrieved items
- swapping retrieval logic later if needed

Do not hard-code retrieval assumptions too deeply into other modules.

---

## Reasoning / Generation Expectations

The model should not be treated as a generic chatbot.
It should be prompted to reason over retrieved evidence and return outputs in a controlled format.

The system should support outputs such as:
- direct answer text
- structured reasoning representation
- symbolic computation-like output
- parseable formats that can be executed or validated

Prefer parseable and constrained outputs over unconstrained prose.

---

## Deterministic Numeric Execution

Whenever possible, numeric reasoning should be executed deterministically in Python.

This means the model may identify:
- relevant values
- operations
- answer type / unit

And Python may then:
- perform arithmetic
- normalize numeric formats
- validate outputs

Keep numeric utilities isolated in dedicated modules.

---

## Evaluation Expectations

Evaluation is a major part of the project.

The codebase should make it easy to compute and inspect at least:

- final-answer correctness
- parse success / failure
- execution success / failure

If practical, it should also support:
- reasoning/program comparison
- evidence retrieval comparison against gold support
- categorized error analysis

Prefer evaluation code that is explicit and readable.

---

## Development Workflow Expectations

This project will be built incrementally with AI assistance.

When generating code:

1. scaffold realistic modules and files
2. include docstrings and TODOs where helpful
3. do not pretend unfinished logic is complete
4. prefer minimal working versions over overly ambitious implementations
5. keep interfaces stable and understandable

Each stage should be easy to run and inspect independently before being connected into a full workflow graph.

---

## Repository Structure Expectations

Create a repository structure that supports iterative development and clear separation of concerns.

Preferred directories:

- `configs/`
- `data/`
- `notebooks/`
- `reports/`
- `scripts/`
- `src/`
- `tests/`

Within `src/`, prefer submodules such as:

- `src/data/`
- `src/retrieval/`
- `src/llm/`
- `src/tools/`
- `src/graph/`
- `src/eval/`
- `src/app/`
- `src/utils/`

This structure can be adjusted slightly if needed, but keep the responsibilities clear.

---

## File Scaffolding Expectations

When creating the skeleton, generate realistic starter files such as:

- `README.md`
- `requirements.txt`
- `.gitignore`
- `.env.example`
- `configs/*.yaml` or similar config files
- source package files under `src/`
- test files under `tests/`
- placeholder report files in `reports/`
- utility scripts under `scripts/`

Starter files should:
- be minimal
- be honest about TODOs
- not contain fake implementations
- give the developer a clear place to continue work

---

## Coding Style

Use:

- Python 3.12-friendly code
- type hints where useful
- dataclasses or pydantic models where helpful
- simple logging
- clean separation between pure logic and I/O
- cross-platform-safe path handling where possible

Prefer:
- `pathlib`
- explicit imports
- small focused functions
- clear naming

Avoid:
- hidden global state
- overly magical decorators
- complex metaprogramming
- framework-heavy patterns unless justified

---

## Environment Assumptions

Assume the developer is working locally:
- VSCode + Git Bash or Powershell + Linux Mint

The code should stay as cross-platform-friendly as practical, except where model serving tools may naturally be Linux-oriented.

Use a standard local Python virtual environment and `requirements.txt`.

Do not assume Poetry, Pipenv, or Conda unless explicitly requested.

---

## What Good Generated Code Looks Like

Good code for this repo should:

- reflect the actual FinQA task
- make intermediate stages visible and testable
- be understandable by a developer new to practical LLM systems
- help the developer explain the architecture clearly later
- support incremental refinement without rewrites

When in doubt, choose the simpler and more explainable design.

---

## Immediate Goal When Scaffolding

The immediate goal is to create a **clean project skeleton** that supports:

- loading FinQA data
- representing evidence units
- retrieval
- LLM prompting / parsing
- numeric tools
- workflow orchestration
- evaluation
- tests
- scripts
- documentation

The skeleton should be ready for incremental vibecoding of each stage.

Do not try to fully implement the whole system in one pass unless explicitly instructed.

Note: PROJECT_BRIEF.md contains a high level briefing for this project.








## Environment Assumptions

Primary development environment:
- OS: Windows 11
- Python: 3.12
- GPU: NVIDIA RTX 4090 Laptop GPU
- Machine: Lenovo ThinkPad laptop
- Editor: Visual Studio Code (latest)
- Shell: Git Bash
- Version control: Git

Development should assume:
- local development on Windows first
- commands should work in Git Bash where possible
- Python tooling should be compatible with Python 3.12
- paths should be handled in a cross-platform-safe way when practical
- the project will be run locally rather than on a remote Linux server during initial development

LLM serving assumptions:
- use an open-source Hugging Face model
- serve locally with vLLM if feasible in the Windows-based workflow
- if vLLM setup is awkward on native Windows, it is acceptable to structure the repo so model serving can be swapped or run in a compatible environment later, while preserving the intended architecture

Implementation preferences based on environment:
- prefer simple local scripts over complex orchestration
- avoid Linux-only assumptions unless clearly isolated in scripts or documentation
- document any GPU/model-size assumptions clearly
- keep setup instructions explicit and reproducible
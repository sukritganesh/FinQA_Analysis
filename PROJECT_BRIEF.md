# FinQA Project Context

This project is a take-home assignment focused on building a practical, explainable question-answering system for the **FinQA** dataset. The user has about **3 years of coding experience** and is comfortable building software, but is **new to AI / practical LLM application development**. The goal of this project is therefore twofold: (1) complete the assignment well, and (2) learn the relevant AI engineering concepts while building it.

## Assignment Context

The assignment is to build a QA chatbot / system for the **FinQA** dataset using:
- **Python**
- an **open-source Hugging Face model**
- local GPU inference
- meaningful use of **vLLM**
- meaningful use of **LangChain**
- meaningful use of **LangGraph**

The assignment also expects more than just working code. It implicitly asks for:
- understanding the dataset
- selecting and justifying a method
- evaluating the system credibly
- discussing production-style considerations
- presenting decisions clearly in a report / presentation

This means the project is not “just build a chatbot.” It is really:
1. build a working financial QA system
2. justify why it is designed that way
3. evaluate it in a numerically meaningful way
4. show engineering judgment and explainability

## User / Environment Context

The user is developing this with heavy AI assistance and wants to “vibecode” the project in an iterative, structured way.

Relevant environment assumptions:
- primary coding experience: solid general software skills, limited practical AI/LLM systems experience
- editor: **VS Code**
- coding style preference: simple, readable, modular Python
- dependency preference: **requirements.txt**, local virtual environment
- avoid Poetry / Pipenv / Conda unless explicitly needed

Local hardware discussed:
- Linux Mint machine
- NVIDIA **RTX 5080**
- around **32 GB RAM** on Linux box
- another machine with a 4090-class GPU was also discussed, but the Linux Mint + 5080 setup is the main target for local serving

The project should stay as cross-platform-friendly as practical, but GPU-heavy local serving will likely happen on Linux.

## Overall Technical Direction

The preferred technical direction is **not** to start with fine-tuning. The first implementation should be **inference only**, using:
- retrieval
- prompt engineering
- controlled output formatting
- deterministic numeric execution where possible
- evaluation against gold labels

The preferred high-level architecture is a **retrieval + reasoning/generation pipeline**, not a pure one-shot prompt over the entire context and not a full autonomous agent.

The working idea is:

1. **Load FinQA data**
2. **Normalize each example into internal structures**
3. **Construct candidate evidence units**
4. **Retrieve relevant evidence for the question**
5. **Pass retrieved evidence into a local LLM**
6. **Have the model produce either a direct answer or a structured reasoning output**
7. **Optionally execute arithmetic deterministically in Python**
8. **Validate / normalize the result**
9. **Compare against gold annotations**

The goal is a modular, explainable system where each stage can be tested independently.

## FinQA Dataset Understanding

FinQA is a dataset for **financial numerical reasoning** over mixed evidence from:
- text before a table (`pre_text`)
- a table (`table`)
- text after a table (`post_text`)
- a question
- gold labels / annotations inside a `qa` object

Important original JSON structure:
- `id`
- `pre_text`
- `post_text`
- `table`
- `qa.question`
- `qa.program`
- `qa.program_re`
- `qa.gold_inds`
- `qa.exe_ans`

Interpretation:
- **runtime / inference inputs**:
  - `id`
  - `pre_text`
  - `post_text`
  - `table`
  - `qa.question`
- **gold labels for evaluation**:
  - `qa.program`
  - `qa.program_re`
  - `qa.gold_inds`
  - `qa.exe_ans`

This separation is important and should be preserved in the codebase.

The dataset provides:
- `train.json`
- `dev.json`
- `test.json`
- `private_test.json`

`dev.json` is the validation/development split.

FinQA is different from ordinary QA because it often requires:
- finding the right sentence / row / year / metric
- combining text and table evidence
- performing arithmetic
- handling units / percentages / financial formatting
- avoiding plausible but numerically wrong answers

This means the project should be designed around **evidence selection + reasoning + numeric correctness**, not just natural-language answer generation.

## Internal Representation Goals

A good normalized internal example structure should clearly separate inference inputs from gold targets.

For example, something like:

```python
{
    "id": "...",
    "question": "...",
    "text_evidence": [...],
    "table_evidence": [...],
    "gold_answer": ...,
    "gold_program": ...,
    "gold_program_re": ...,
    "gold_support": ...
}
```

This is not necessarily the raw dataset format, but it is a useful internal representation for the project.

## Retrieval Strategy

The preferred first retrieval approach is **not LLM-based retrieval**.

Instead, the current preferred direction is:
- **BM25 over sentence / row candidates**
- plus **hand-built boosts / heuristics**

Candidate evidence units should likely include:
- individual sentences from `pre_text`
- individual sentences from `post_text`
- table rows rendered into text-like form

The user and assistant discussed that FinQA is a strong fit for this because many questions depend on:
- exact metric names
- exact years
- exact table rows
- exact numeric fields

Suggested initial retrieval design:
- BM25 scoring over candidate evidence units
- plus hand-built boosts for things like:
  - year overlap
  - metric keyword matches
  - comparison words like increase / decrease / difference
  - percentage-related cues
  - numeric density
  - possibly mild type preference (table row vs text sentence)

A future improvement may combine **BM25 + vector retrieval** as hybrid retrieval, but the first version should be simpler and highly inspectable.

LLM-based retrieval or reranking was discussed as possible, but not preferred for V1. A possible later version could use:
- BM25 to get top candidates
- LLM reranking over a small candidate set

But the first version should keep retrieval classical, cheap, fast, and interpretable.

## Generator / Reasoning Stage

The retrieved evidence should be passed into a local LLM.

The generator’s job is not just to “chat.” It should reason over the evidence and produce one of:
- a direct final answer
- a structured reasoning output
- a symbolic program-like output
- another parseable constrained format

The preferred design is to constrain outputs carefully so they are parseable.

The model should **not** just be prompted loosely and allowed to answer however it wants. Prompting should specify:
- what evidence is provided
- what format to output
- whether to return a final answer, a structured object, or a symbolic computation
- avoid extra prose if the output needs to be parsed

It was explicitly discussed that **LangChain** is useful here for:
- prompt templates
- structured output schemas
- model invocation wrappers
- output parsing
- formatting retrieved evidence for the prompt

The project should avoid unconstrained messy outputs whenever possible.

## Deterministic Numeric Execution

A key design principle is:
- let the model identify values and operations
- let Python perform the final math whenever practical

This is preferable for:
- numerical reliability
- reproducibility
- explainability
- evaluation clarity

The user’s vibecoded skeleton already includes utility functions for arithmetic, and those are expected to be part of the solution.

So a strong V1 system may look like:
- retrieve evidence
- prompt model for structured reasoning output
- parse that output
- execute the computation in Python
- normalize the result
- compare to gold answer

## Evaluation Strategy

Evaluation is a major part of the project.

The system should not only say “here’s an answer.” It should also support evaluation of:
- final answer correctness
- parse success / failure
- execution success / failure
- evidence retrieval quality (where practical)
- reasoning/program similarity (where practical)

Good evaluation targets:
- `qa.exe_ans` as the gold final answer
- `qa.program` and/or `qa.program_re` for reasoning comparison
- `qa.gold_inds` for retrieval comparison

This should enable stage-by-stage debugging:
- retrieval failure
- wrong row / year / metric selected
- malformed structured output
- arithmetic error
- formatting mismatch
- validation failure

The project should make these failure modes visible.

## Role of vLLM, LangChain, and LangGraph

The conceptual stack has already been worked out:

### vLLM
vLLM is the **model serving / inference layer**.
It is how the chosen Hugging Face model is run locally on GPU and exposed through a clean interface, likely as a local API server.

Mental model:
- Hugging Face provides the model weights
- vLLM runs the model efficiently
- the rest of the application calls the model through vLLM

This should be treated as the local serving layer, not as business logic.

### LangChain
LangChain is the **application building-block layer**.

It should be used for:
- prompt templates
- model wrappers / invocation
- structured output definitions
- output parsing
- formatting retrieved evidence into prompts
- possibly tool wrappers such as calculator integration

It should not be overused for its own sake. Use it where it helps make the code cleaner and more modular.

### LangGraph
LangGraph is the **workflow / orchestration layer**.

It should be used to define the high-level stateful pipeline, such as:
- load / prepare example
- retrieve evidence
- generate reasoning output
- execute calculation
- validate
- format final answer

LangGraph is the high-level workflow engine, not the place for complex logic inside every step.

A key design idea discussed:
- individual steps should first be built and tested as ordinary functions
- once stable, they should be wrapped as LangGraph nodes
- the graph should be assembled incrementally, not all at once at the very end

## Model Choice Context

The current strongest discussed model choice for this project is:
- **`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`**

Why:
- local reasoning-heavy model
- realistic to run locally on the discussed hardware
- suitable for a reasoning-oriented task like FinQA
- should be serveable via vLLM on Linux

Safer fallback:
- **`Qwen/Qwen2.5-14B-Instruct`**

The first version of the codebase should not hardcode one exact model choice too deeply; keep the interface flexible.

## Development Philosophy / Vibecoding Workflow

This project is being intentionally built with heavy AI assistance.

Preferred development loop:
1. let AI generate scaffolding or first-pass code
2. run it immediately
3. inspect real outputs
4. improve the specific stage
5. only then integrate it into the larger pipeline

Important principle:
- each stage is its own mini project at first

That means:
- vibecode one stage
- run it in isolation
- debug and revise it
- possibly vibecode a better second version of the same stage
- then move on

Graph integration should be **incremental**:
- build and test stage functions first
- wrap them as nodes once they are stable enough
- connect a few nodes
- test the partial graph
- add the next node
- repeat

Do **not** try to build the entire final graph in a single shot before testing the components.

## Current Scaffolding / Repo Intent

The repo skeleton is already being scaffolded / vibecoded, and the project should support at least these concerns:
- data loading
- schemas / types
- evidence construction
- retrieval
- LLM prompting / parsing
- numeric tools
- workflow state / graph
- evaluation
- CLI or simple demo entrypoint
- tests
- scripts
- reports / presentation placeholders

Preferred directory pattern:
- `configs/`
- `data/`
- `notebooks/`
- `reports/`
- `scripts/`
- `src/`
- `tests/`

Likely `src/` subpackages:
- `src/data/`
- `src/retrieval/`
- `src/llm/`
- `src/tools/`
- `src/graph/`
- `src/eval/`
- `src/app/`
- `src/utils/`

The code should be:
- simple
- modular
- readable
- lightly typed
- easy to explain
- not overengineered

Avoid:
- giant abstractions
- speculative enterprise architecture
- unnecessary autonomous agent logic
- premature complexity

## Immediate Goal

The immediate goal is to create and then incrementally fill in a clean scaffold that supports:
- loading FinQA examples
- representing evidence units
- retrieval over those units
- local LLM calls through vLLM
- prompt / output formatting through LangChain
- deterministic math utilities
- LangGraph-based workflow orchestration
- evaluation against gold FinQA labels

The first version is **inference only**:
- no fine-tuning
- no advanced agent behavior
- no heavy UI
- no distributed deployment

## Overall Success Criterion

A successful project should:
- load FinQA correctly
- transform examples into usable evidence units
- retrieve the right rows / sentences reasonably well
- prompt a local LLM with selected evidence
- get parseable outputs in a controlled format
- execute arithmetic when appropriate
- produce a final answer that can be evaluated
- expose enough intermediate information to debug and explain the system
- serve as a strong take-home submission that demonstrates both working code and thoughtful engineering judgment

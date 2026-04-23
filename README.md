# llm-eval-harness

A lightweight CLI + library for evaluating LLM application outputs.

Feed it a set of test inputs and an expected-output schema. It runs every input
through a **target LLM**, then scores each response with a **judge LLM** —
returning per-case and aggregate scores for correctness, relevance, and format
compliance.

---

## Features

- **Target LLMs**: OpenAI and Anthropic (configurable per run)
- **Judge LLM**: Any OpenAI or Anthropic model, independent of the target
- **Structured scores**: `correctness`, `relevance`, `format_compliance` (0–1) + reasoning
- **Rich console report**: per-case table + aggregate summary
- **Suite format**: self-contained test suite files with embedded prompt template and output schema
- **Extensible**: separate `runners/`, `judges/`, `reporters/` modules

---

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python ≥ 3.11.

---

## Quick start

Set your API key(s):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# and/or
export OPENAI_API_KEY=sk-...
```

Run the bundled suite (uses Anthropic Haiku for both target and judge):

```bash
llm-eval run \
  --inputs examples/test_inputs.json \
  --provider anthropic \
  --model    claude-haiku-4-5-20251001
```

Use a stronger judge model for more reliable scoring:

```bash
llm-eval run \
  --inputs       examples/test_inputs.json \
  --provider     anthropic \
  --model        claude-haiku-4-5-20251001 \
  --judge-provider anthropic \
  --judge-model    claude-sonnet-4-5-20251001
```

Run with OpenAI models:

```bash
llm-eval run \
  --inputs       examples/test_inputs.json \
  --provider     openai \
  --model        gpt-4o-mini \
  --judge-provider openai \
  --judge-model    gpt-4o-mini
```

---

## CLI reference

```
llm-eval run [OPTIONS]

Options:
  -i, --inputs PATH            JSON file: EvalSuite object or list of EvalInput  [required]
  -t, --template PATH          Prompt template file — overrides suite's prompt_template
  -s, --schema PATH            JSON Schema file — overrides suite's output_schema
      --provider TEXT          Target LLM provider: anthropic | openai  [default: anthropic]
  -m, --model TEXT             Target LLM model name
      --judge-provider TEXT    Judge LLM provider: anthropic | openai  [default: anthropic]
      --judge-model TEXT       Judge LLM model name
      --max-tokens INT         Max tokens for target LLM responses  [default: 1024]
```

`--template` and `--schema` are optional when the inputs file is an `EvalSuite`
(see below) — the suite's embedded values are used automatically and can be
overridden per-run with these flags.

---

## Input file formats

### Suite format (recommended)

A single self-contained JSON object that bundles the prompt template, output
schema, and all test cases together:

```json
{
  "suite_name": "my_suite",
  "description": "Optional human-readable description.",
  "prompt_template": "{question}",
  "output_schema": {
    "type": "string",
    "max_length": 300,
    "format": "plain text, no markdown, no preamble"
  },
  "cases": [
    {
      "id": "unique-case-id",
      "question": "What is the capital of France?",
      "expected": "Paris"
    }
  ]
}
```

- `suite_name` — identifier printed at run time
- `description` — free-text description (optional)
- `prompt_template` — prompt with `{question}` placeholder (can be overridden with `--template`)
- `output_schema` — describes the expected response shape (can be overridden with `--schema`)
- `cases[].id` — unique identifier shown in the report
- `cases[].question` — substituted into `{question}` in the template
- `cases[].expected` — optional reference answer used by the judge

### Legacy list format

`--inputs` can also be a JSON array of `EvalInput` objects. In this case
`--template` and `--schema` are **required**:

```json
[
  {
    "id": "unique-case-id",
    "variables": { "question": "What is 2 + 2?" },
    "expected_output": "4"
  }
]
```

- `variables` — key/value pairs substituted into the prompt template via `{key}`
- `expected_output` — optional reference answer used by the judge

---

## RAG evaluation (v0.2)

The `eval-rag` command evaluates a Retrieval-Augmented Generation pipeline end-to-end.

### RAG metrics

| Metric | What it measures |
|---|---|
| **Context Relevance** | Are the retrieved chunks relevant to the question? A low score means retrieval is pulling in noisy or unrelated documents. |
| **Faithfulness** | Is the answer grounded in the retrieved context, or does the model hallucinate facts not present there? A low score is a hallucination signal. |
| **Answer Relevance** | Does the answer actually address the question asked? A low score means the model went off-topic even if the context was correct. |

All three scores are 0–1 floats produced by a judge LLM call with structured JSON output. Each dimension also includes a one-to-two sentence reasoning string.

### Running the example suite

The example loads a small fictional "VectorDB Lite" documentation corpus (4 markdown files) and runs 6 test cases that exercise:
- Direct single-chunk retrieval
- Cross-document fact synthesis (requires combining two chunks)
- Out-of-corpus questions (faithfulness stress test — model should say it doesn't know)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export VOYAGE_API_KEY=pa-...        # required for embeddings

llm-eval eval-rag \
  --corpus-dir examples/rag_suite/corpus \
  --suite      examples/rag_suite/rag_suite.json \
  --provider   anthropic \
  --model      claude-haiku-4-5-20251001 \
  --judge-provider anthropic \
  --judge-model    claude-haiku-4-5-20251001 \
  --top-k 3
```

Save retrieved chunks to a file for debugging:

```bash
llm-eval eval-rag \
  --corpus-dir examples/rag_suite/corpus \
  --suite      examples/rag_suite/rag_suite.json \
  --save-retrieved-chunks retrieved_debug.json
```

### Full `eval-rag` options

```
llm-eval eval-rag [OPTIONS]

Options:
  -c, --corpus-dir PATH        Directory containing .txt/.md corpus files  [required]
  -s, --suite PATH             Path to rag_suite.json (RAGSuite format)  [required]
  -m, --model TEXT             Target LLM model name  [default: claude-haiku-4-5-20251001]
      --provider TEXT          Target LLM provider: anthropic | openai  [default: anthropic]
      --judge-provider TEXT    Judge LLM provider: anthropic | openai  [default: anthropic]
      --judge-model TEXT       Judge LLM model name
      --top-k INT              Chunks to retrieve per question  [default: 3]
      --chunk-size INT         Chunk size in approximate tokens  [default: 256]
      --chunk-overlap INT      Overlap between chunks in tokens  [default: 32]
      --embedding-model TEXT   Voyage AI embedding model  [default: voyage-3]
      --max-tokens INT         Max tokens for target LLM responses  [default: 1024]
      --save-retrieved-chunks PATH  Save retrieved chunks per case to a JSON file
```

### RAG suite format

A `rag_suite.json` file is a JSON object:

```json
{
  "suite_name": "my_rag_suite",
  "description": "Optional description.",
  "cases": [
    {
      "id": "unique-case-id",
      "question": "What is the default similarity metric?",
      "expected": "cosine",
      "notes": "Optional note for humans reviewing results."
    }
  ]
}
```

- `expected` is optional. When provided, the judge uses it as a reference answer to assess faithfulness and correctness.
- `notes` is purely informational and not sent to the judge.

---

## Project structure

```
llm_eval_harness/
├── cli.py               # typer CLI — `run` and `eval-rag` commands
├── models.py            # pydantic schemas (all v0.1 + v0.2 RAG models)
├── corpus.py            # document loader + overlapping chunker
├── embedder.py          # Voyage AI embedder + in-memory VectorStore
├── rag_runner.py        # retrieval + generation orchestration
├── rag_judge.py         # RAG-specific LLM judge (context_relevance, faithfulness, answer_relevance)
├── runners/
│   ├── base.py
│   ├── anthropic_runner.py
│   └── openai_runner.py
├── judges/
│   ├── base.py
│   └── llm_judge.py
└── reporters/
    └── console.py       # rich tables for both standard and RAG results
```

---

## Extending the harness

**Add a new runner** — subclass `BaseRunner` in `llm_eval_harness/runners/`:

```python
from llm_eval_harness.runners.base import BaseRunner

class MyRunner(BaseRunner):
    def run(self, prompt: str) -> str:
        ...
```

**Add a new judge** — subclass `BaseJudge` in `llm_eval_harness/judges/`:

```python
from llm_eval_harness.judges.base import BaseJudge
from llm_eval_harness.models import JudgeScore

class MyJudge(BaseJudge):
    def score(self, prompt, output, expected_output, output_schema) -> JudgeScore:
        ...
```

**Swap in a real vector DB** — `VectorStore` in `embedder.py` exposes `add(chunks)` and `retrieve(query_embedding, top_k)`. Implement the same interface backed by Pinecone, Weaviate, pgvector, etc.

**Add a new reporter** — any callable that accepts `list[EvalResult]` or `list[RAGResult]` works.

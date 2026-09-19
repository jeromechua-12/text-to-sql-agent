# Text-to-SQL Agent

Turns a natural-language question into SQL, runs it against a SQLite database, and
retries with structured feedback when it fails. The value is the runtime harness
around the model: the agent loop, the guardrail, tracing, and structured logging.

## How a request flows

A question and a database id go in. The agent retrieves the relevant schema,
generates SQL, checks it against the guardrail, executes it in a read-only sandbox,
and on failure packages the error into feedback and loops back. It stops on the
first working query or after the attempt limit, then emits one structured record.

## Components

**Agent loop** (`agent.py`)
An explicit LangGraph state machine: retrieve -> generate -> guardrail -> execute
-> observe. On a rejected or failed attempt, `observe` packages the previous SQL,
the stage that failed, the raw error, and a corrective hint into an `Observation`
that is added to the next generation prompt. Loops until success or `max_attempts`.

**Guardrail** (`guardrail.py`)
Static check between generation and execution. Parses the SQL with sqlglot and
allows it only if it is a single read-only SELECT. Rejects writes, DDL, multiple
statements, PRAGMA, and ATTACH. Returns a structured allow/reject verdict with a
reason, which is logged and fed back into the loop on rejection.

**Sandbox** (`sandbox.py`)
Executes queries over a SQLite connection opened read-only (`mode=ro` plus
`PRAGMA query_only`). Returns rows on success or the raw database error as data.
Also renders full-schema DDL as the fallback when no retriever is configured.

**Schema retrieval** (`schema_retrieval.py`)
Embeds one description per table and per column with sentence-transformers, indexes
them in FAISS, and keeps only the top-k tables a question needs instead of dumping
the whole schema into the prompt. Each database is indexed once and reused.

**LLM interface** (`llm.py`)
A one-method `complete(system, user)` protocol with an OpenAI implementation, so the
model is swappable. Reports token usage, latency, and cost per call.

**Run log** (`run_log.py`)
One versioned pydantic record per run capturing the question, retrieved schema,
every attempt's SQL and guardrail verdict, execution result or error, retry and
recovery counts, tokens, cost, and latency. Appended as JSON lines. This is the
schema the tracing and evaluation read from.

**Tracing** (`tracing.py`)
Langfuse wired over every node: one trace per run with a child span per node
(retrieve, generate, guardrail, execute, observe), carrying timings, token counts,
and verdicts so each run has a visual trajectory.

**API** (`api.py`)
FastAPI serving layer. `POST /ask` runs the agent for a question against a Spider
database id and returns the structured run record; `GET /databases` lists available
ids; `GET /health` reports liveness and whether retrieval is active.

**Smoke test** (`smoke_test.py`)
Runs the agent over a small Spider dev subset and reports execution accuracy,
self-correction recovery rate, and median/p95 latency. Enough to prove the harness
works end to end, not a benchmark. Flags:

- `--n` number of dev questions to run (default 40)
- `--dev-json` path to the Spider dev JSON
- `--db-root` root directory of the Spider database files
- `--output-dir` where the run logs and summary are written
- `--model` LLM model id (default `gpt-4o-mini`)
- `--max-attempts` retry budget per question (default 3)
- `--embedding-model` sentence-transformers model for retrieval
- `--top-k` tables kept in the prompt per question
- `--full-schema` skip retrieval and put every table in the prompt

## Running

Managed with uv, Python 3.13.

```
uv sync
uv run uvicorn text_to_sql_agent.api:app     # serve the API
uv run python -m text_to_sql_agent.smoke_test # run the smoke test
uv run pytest                                  # tests
```

Set `OPENAI_API_KEY` (and Langfuse keys for tracing) in `.env`. The API can also
be run with Docker via `docker compose up`.

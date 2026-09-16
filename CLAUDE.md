# CLAUDE.md

Guidance for Claude when acting as a coding assistant on this project. Read this before writing or editing code.

## Objective

Build a text-to-SQL agent as an MVP whose value is the runtime harness around the model, not the model itself. Given a natural-language question and a target database, the agent retrieves the relevant part of the schema, generates SQL, passes it through a guardrail, executes it in a read-only sandbox, and on failure packages the error into structured feedback and retries within a bounded attempt limit.

The focus of this project is the engineering around the LLM: the agent loop and feedback loop, the guardrail layer, tracing, and structured logging. Evaluation is intentionally kept small — a smoke test that proves the system works end to end and produces a rough number. This is an MVP to showcase runtime-system skills, not a production-hardened or benchmark-chasing agent.

Priorities, in order: a clean and inspectable runtime harness first, then agent capability, then the demo and polish. When a tradeoff arises, favour whatever keeps the system clear, safe, and observable.

## Main Components

Agent loop and feedback loop: an explicit state machine — generate SQL, validate, execute, observe result, and on error package the raw database error into a structured observation (previous query, previous error, attempt count) and loop back, else finish. Implement the control flow explicitly in LangGraph so every node is a visible, instrumented step. The interesting engineering is the feedback packaging, not just the retry.

Guardrail layer: a middleware interceptor between generation and execution with a clear allow/reject contract. Every query is parsed with sqlglot and must be a single SELECT; reject writes, multiple statements, PRAGMA, and ATTACH. Execution runs over a read-only connection. Rejections carry a reason, are logged, and can feed back into the correction loop.

Tracing layer: Langfuse wired over every node so each run produces a visual trajectory — spans for generation, guardrail check, execution, and each retry, with timings and token counts nested underneath.

Observability and structured logging: every run emits one structured record (pydantic schema) capturing the question, retrieved schema, each attempt's generated SQL, guardrail verdict, execution result or error, retry count, tokens, cost, and latency. This is the backbone the tracing and any later analysis read from. Design the log schema deliberately so it is inspectable and extensible.

Schema retrieval: lightweight schema-linking so the agent is not a toy — embed and retrieve the relevant tables and columns rather than dumping the full schema into the prompt. A component, not the focus.

Serving and demo: a FastAPI endpoint as the real interface, and an optional Streamlit "ask your database" page as a thin demo layer for showing the system working.

## Evaluation Metrics

Keep evaluation small — a smoke test over roughly 30 to 50 Spider questions, enough to prove the harness works end to end and to have a number. Report:

- Execution accuracy: run the generated SQL and compare result sets against the gold answer.
- Self-correction recovery rate: the share of first-attempt failures the retry loop rescued.
- Latency: end-to-end latency per query, including retries, reported as median and p95. Log per-node timings via the tracing layer so slow steps are visible.

No ablation tables, no judge validation, and no full failure taxonomy at this stage; those belong to a later evaluation phase.

## Tech Stack

Language and tooling: Python 3.13.13, managed with uv for the virtual environment and dependencies. Use uv for running the project and adding packages; do not fall back to bare pip or poetry.

Database: SQLite. The Spider databases ship as SQLite files, so no database server is required. Open all execution connections in read-only mode.

Dataset: the Spider dataset. Work against a subset of the dev set for the smoke test rather than the full data.

Agent orchestration: LangGraph, with the generate/validate/execute/correct control flow written explicitly.

LLM: a low-cost model (for example GPT-4o-mini or Claude Haiku) accessed through a single swappable interface.

Schema retrieval: sentence-transformers with a FAISS index over serialised table and column descriptions.

SQL safety and parsing: sqlglot for parsing and validation, plus a read-only SQLite connection.

Tracing: Langfuse for agent trajectory inspection.

Serving: FastAPI, containerised with Docker. Optional Streamlit interface for the demo.

Testing: pytest for the guardrail contract and the loop.

## Code Style

Write direct implementations. No speculative factory classes or abstract base classes for single-use logic.

No comments except for function and method docstrings.

Keep docstrings short, maximum of 2 lines (160 chars), stating what it does.

Use type hints for function arguments and return values only.

No redundant try/except error handling.

## Instructions

Keep the guardrail, agent loop, tracing, and logging as separate, independently testable modules.

Each feature should be written in its own branch/worktree.

Never widen the SQL sandbox permissions for convenience — read-only and SELECT-only are non-negotiable.

Design the log schema early and richly, since the tracing layer read from it.

Prefer small, verifiable changes and state which component each change affects.

You are not allowed to run git or bash commands without permission.

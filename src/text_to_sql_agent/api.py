"""FastAPI serving layer: the real interface over the agent, resolving a Spider database by id and returning the structured run record."""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from text_to_sql_agent.agent import AgentRun, run_agent
from text_to_sql_agent.llm import LLM, OpenAIChat
from text_to_sql_agent.run_log import AttemptRecord, RetrievalRecord, RunLogger, RunRecord
from text_to_sql_agent.schema_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    SchemaRetriever,
    SentenceTransformerEmbedder,
)
from text_to_sql_agent.tracing import Tracer

DEFAULT_DB_ROOT = Path("data/spider_data/database")
DEFAULT_RUN_LOG_PATH = Path("logs/api/runs.jsonl")
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class Dependencies:
    """The long-lived objects the endpoints share, built once at startup or injected in tests."""

    llm: LLM
    db_root: Path
    run_logger: RunLogger
    tracer: Tracer
    retriever: SchemaRetriever | None


class AskRequest(BaseModel):
    """A natural-language question against one Spider database, with the retry budget for the loop."""

    question: str
    db_id: str
    max_attempts: int = 3


class QueryResult(BaseModel):
    """The result set of the winning query, capped at the sandbox row limit."""

    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool


class AskResponse(BaseModel):
    """The structured outcome of one run: the final SQL and its rows, plus every attempt and the usage summary."""

    run_id: str
    status: Literal["success", "failed"]
    final_sql: str | None
    result: QueryResult | None
    retrieval: RetrievalRecord | None
    attempts: list[AttemptRecord]
    retry_count: int
    recovered: bool
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    trace_id: str | None


def build_dependencies() -> Dependencies:
    """Construct the real dependencies from environment, loading the embedder once unless FULL_SCHEMA is set."""
    load_dotenv()
    db_root = Path(os.getenv("SPIDER_DB_ROOT", str(DEFAULT_DB_ROOT)))
    model = os.getenv("AGENT_MODEL", DEFAULT_MODEL)
    run_log_path = Path(os.getenv("RUN_LOG_PATH", str(DEFAULT_RUN_LOG_PATH)))
    full_schema = os.getenv("FULL_SCHEMA", "").lower() in {"1", "true", "yes"}
    top_k = int(os.getenv("TOP_K", str(DEFAULT_TOP_K)))
    embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    retriever = None if full_schema else SchemaRetriever(SentenceTransformerEmbedder(embedding_model), top_k=top_k)
    return Dependencies(OpenAIChat(model), db_root, RunLogger(run_log_path), Tracer(), retriever)


def db_path_for(db_root: Path, db_id: str) -> Path:
    """Locate the SQLite file for a Spider database id under the configured database root."""
    return db_root / db_id / f"{db_id}.sqlite"


def build_response(run: AgentRun, record: RunRecord) -> AskResponse:
    """Assemble the API response from the finished run and its structured log record."""
    execution = run.result
    result = (
        QueryResult(
            columns=execution.columns,
            rows=[list(row) for row in execution.rows],
            row_count=len(execution.rows),
            truncated=execution.truncated,
        )
        if execution is not None
        else None
    )
    return AskResponse(
        run_id=record.run_id,
        status=record.status,
        final_sql=record.final_sql,
        result=result,
        retrieval=record.retrieval,
        attempts=record.attempts,
        retry_count=record.retry_count,
        recovered=record.recovered,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        cost_usd=record.cost_usd,
        latency_ms=record.latency_ms,
        trace_id=record.trace_id,
    )


def create_app(dependencies: Dependencies | None = None) -> FastAPI:
    """Build the FastAPI app, wiring real dependencies at startup unless a set is injected for testing."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.deps = dependencies if dependencies is not None else build_dependencies()
        yield

    app = FastAPI(title="Text-to-SQL Agent", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        """Report liveness and whether schema retrieval is active."""
        deps: Dependencies = request.app.state.deps
        return {"status": "ok", "retrieval": deps.retriever is not None}

    @app.get("/databases")
    def databases(request: Request) -> dict:
        """List the database ids available under the configured database root."""
        deps: Dependencies = request.app.state.deps
        ids = sorted(path.name for path in deps.db_root.glob("*") if db_path_for(deps.db_root, path.name).exists())
        return {"databases": ids}

    @app.post("/ask")
    def ask(request: Request, body: AskRequest) -> AskResponse:
        """Run the agent for one question against the named database and return the structured run record."""
        deps: Dependencies = request.app.state.deps
        db_path = db_path_for(deps.db_root, body.db_id)
        if not db_path.exists():
            raise HTTPException(status_code=404, detail=f"Unknown database '{body.db_id}'.")
        run = run_agent(
            body.question,
            db_path,
            deps.llm,
            retriever=deps.retriever,
            max_attempts=body.max_attempts,
            tracer=deps.tracer,
        )
        record = deps.run_logger.log(run, source="api", db_id=body.db_id)
        return build_response(run, record)

    return app


app = create_app()

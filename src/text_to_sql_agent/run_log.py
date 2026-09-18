"""Structured run logging: one versioned pydantic record per run, appended as JSON lines and summarised to stdlib logging."""

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from text_to_sql_agent.agent import AgentRun, Attempt
from text_to_sql_agent.guardrail import GuardrailDecision, RejectionReason
from text_to_sql_agent.schema_retrieval import TableMatch

SCHEMA_VERSION = 2
ROWS_PREVIEW_LIMIT = 5

logger = logging.getLogger("text_to_sql_agent.run_log")

AttemptOutcome = Literal["success", "guardrail_rejected", "execution_error"]


class RetrievalRecord(BaseModel):
    """Which tables schema retrieval kept for the prompt, ranked with scores, out of how many the database has."""

    total_tables: int
    top_k: int
    tables: list[TableMatch]


class GuardrailRecord(BaseModel):
    """Verdict of the guardrail check for one attempt."""

    decision: GuardrailDecision
    reason_code: RejectionReason | None = None
    reason: str | None = None


class ExecutionRecord(BaseModel):
    """Outcome of running one attempt in the sandbox; rows are summarised rather than stored in full."""

    ok: bool
    columns: list[str] = []
    row_count: int = 0
    rows_preview: list[list[Any]] = []
    truncated: bool = False
    error: str | None = None
    latency_ms: float = 0.0


class ObservationRecord(BaseModel):
    """Feedback packaged from a failed attempt and shown to the model on the next attempt."""

    stage: Literal["guardrail", "execution"]
    error: str
    hint: str


class AttemptRecord(BaseModel):
    """One pass through generate -> guardrail -> execute, with usage and the verdict at each stage."""

    index: int
    outcome: AttemptOutcome
    sql: str
    raw_response: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    llm_latency_ms: float
    guardrail: GuardrailRecord | None = None
    execution: ExecutionRecord | None = None
    observation: ObservationRecord | None = None


class RunRecord(BaseModel):
    """The single structured record emitted per run; the source the tracing layer and evaluation read from."""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    recorded_at: datetime
    question: str
    db_path: str
    schema_text: str
    retrieval: RetrievalRecord | None = None
    model: str
    status: Literal["success", "failed"]
    final_sql: str | None
    attempt_count: int
    retry_count: int
    recovered: bool
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    attempts: list[AttemptRecord]
    trace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_run(
        cls,
        run: AgentRun,
        run_id: str | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "RunRecord":
        """Build the log record for a finished run, deriving the retry and recovery summary fields."""
        attempt_count = len(run.attempts)
        retrieval = run.retrieval
        return cls(
            run_id=run_id or uuid.uuid4().hex,
            recorded_at=datetime.now(UTC),
            question=run.question,
            db_path=run.db_path,
            schema_text=run.schema_text,
            retrieval=(
                RetrievalRecord(total_tables=retrieval.total_tables, top_k=retrieval.top_k, tables=retrieval.tables)
                if retrieval is not None
                else None
            ),
            model=run.attempts[0].llm.model,
            status=run.status,
            final_sql=run.final_sql,
            attempt_count=attempt_count,
            retry_count=attempt_count - 1,
            recovered=run.status == "success" and attempt_count > 1,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            cost_usd=run.cost_usd,
            latency_ms=run.latency_ms,
            attempts=[attempt_record(attempt) for attempt in run.attempts],
            trace_id=trace_id if trace_id is not None else run.trace_id,
            metadata=metadata or {},
        )


def attempt_record(attempt: Attempt) -> AttemptRecord:
    """Flatten one in-memory attempt into its log record."""
    verdict = attempt.verdict
    execution = attempt.execution
    observation = attempt.observation
    return AttemptRecord(
        index=attempt.index,
        outcome=attempt_outcome(attempt),
        sql=attempt.sql,
        raw_response=attempt.raw_response,
        model=attempt.llm.model,
        prompt_tokens=attempt.llm.prompt_tokens,
        completion_tokens=attempt.llm.completion_tokens,
        cost_usd=attempt.llm.cost_usd,
        llm_latency_ms=attempt.llm.latency_ms,
        guardrail=(
            GuardrailRecord(decision=verdict.decision, reason_code=verdict.reason_code, reason=verdict.reason)
            if verdict is not None
            else None
        ),
        execution=(
            ExecutionRecord(
                ok=execution.ok,
                columns=execution.columns,
                row_count=len(execution.rows),
                rows_preview=[list(row) for row in execution.rows[:ROWS_PREVIEW_LIMIT]],
                truncated=execution.truncated,
                error=execution.error,
                latency_ms=execution.latency_ms,
            )
            if execution is not None
            else None
        ),
        observation=(
            ObservationRecord(stage=observation.stage, error=observation.error, hint=observation.hint)
            if observation is not None
            else None
        ),
    )


def attempt_outcome(attempt: Attempt) -> AttemptOutcome:
    """Classify an attempt by the stage that decided it."""
    if attempt.verdict is not None and not attempt.verdict.allowed:
        return "guardrail_rejected"
    if attempt.execution is not None and attempt.execution.ok:
        return "success"
    return "execution_error"


class RunLogger:
    """Appends one RunRecord per run as a JSON line to a file and emits a one-line summary to stdlib logging."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(self, run: AgentRun, trace_id: str | None = None, **metadata: Any) -> RunRecord:
        """Record a finished run and return the record that was written."""
        record = RunRecord.from_run(run, trace_id=trace_id, metadata=metadata)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        logger.info(
            "run %s %s attempts=%d retries=%d recovered=%s tokens=%d cost_usd=%.6f latency_ms=%.0f",
            record.run_id,
            record.status,
            record.attempt_count,
            record.retry_count,
            record.recovered,
            record.prompt_tokens + record.completion_tokens,
            record.cost_usd,
            record.latency_ms,
        )
        return record


def load_records(path: str | Path) -> list[RunRecord]:
    """Read every record back from a JSON lines file written by RunLogger."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [RunRecord.model_validate_json(line) for line in lines if line.strip()]

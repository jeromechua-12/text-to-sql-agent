"""Langfuse tracing: one trace per run with a child observation per graph node, carrying timings, usage, and verdicts."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

from dotenv import load_dotenv
from langfuse import Langfuse

from text_to_sql_agent.guardrail import GuardrailVerdict
from text_to_sql_agent.llm import LLMResponse
from text_to_sql_agent.sandbox import ExecutionResult

TRACE_NAME = "text_to_sql"
ROWS_PREVIEW_LIMIT = 5

StepType = Literal["agent", "generation", "guardrail", "tool", "span"]


class Step:
    """One instrumented node; records the node's outcome onto the underlying Langfuse observation, if any."""

    def __init__(self, observation: Any | None) -> None:
        self.observation = observation

    def record_generation(self, response: LLMResponse) -> None:
        """Attach the model reply, token usage, and cost to a generation observation."""
        self._update(
            output=response.text,
            model=response.model,
            usage_details={
                "input": response.prompt_tokens,
                "output": response.completion_tokens,
                "total": response.prompt_tokens + response.completion_tokens,
            },
            cost_details={"total": response.cost_usd},
        )

    def record_verdict(self, verdict: GuardrailVerdict) -> None:
        """Attach the guardrail decision, flagging a rejection as a warning with its reason."""
        self._update(
            output=verdict.model_dump(mode="json", exclude={"sql"}),
            level="DEFAULT" if verdict.allowed else "WARNING",
            status_message=verdict.reason,
        )

    def record_execution(self, result: ExecutionResult) -> None:
        """Attach a row summary with a short preview on success, or the raw database error flagged as an error."""
        if not result.ok:
            self._update(output={"error": result.error}, level="ERROR", status_message=result.error)
            return
        self._update(
            output={
                "columns": result.columns,
                "row_count": len(result.rows),
                "rows_preview": [list(row) for row in result.rows[:ROWS_PREVIEW_LIMIT]],
                "truncated": result.truncated,
            }
        )

    def record_output(self, output: Any) -> None:
        """Attach an arbitrary JSON-serialisable output to the observation."""
        self._update(output=output)

    def _update(self, **fields: Any) -> None:
        if self.observation is not None:
            self.observation.update(**fields)


class Tracer:
    """Opens one Langfuse trace per run and one child observation per node; a no-op when no Langfuse keys are set."""

    def __init__(self, client: Langfuse | None = None) -> None:
        load_dotenv()
        configured = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
        self.client = client if client is not None else (Langfuse() if configured else None)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @contextmanager
    def run(self, question: str, db_path: str, max_attempts: int) -> Iterator[Step]:
        """Open the root observation of a new trace around one agent run and flush it when the block exits."""
        metadata = {"db_path": db_path, "max_attempts": max_attempts}
        try:
            with self._observe(TRACE_NAME, "agent", {"question": question}, metadata) as root:
                yield root
        finally:
            if self.client is not None:
                self.client.flush()

    @contextmanager
    def step(self, name: str, kind: StepType, attempt: int, inputs: Any) -> Iterator[Step]:
        """Open a child observation for one node of the given attempt, nested under the current trace."""
        with self._observe(name, kind, inputs, {"attempt": attempt}) as step:
            yield step

    def current_trace_id(self) -> str | None:
        """Return the id of the trace open on the current context, or None when tracing is disabled."""
        return self.client.get_current_trace_id() if self.client is not None else None

    @contextmanager
    def _observe(self, name: str, kind: StepType, inputs: Any, metadata: dict[str, Any]) -> Iterator[Step]:
        """Open a Langfuse observation as the current span, or yield an inert step when tracing is disabled."""
        if self.client is None:
            yield Step(None)
            return
        with self.client.start_as_current_observation(name=name, as_type=kind, input=inputs, metadata=metadata) as observation:
            yield Step(observation)

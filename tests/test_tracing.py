import json
import sqlite3

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from text_to_sql_agent.agent import SYSTEM_PROMPT, run_agent
from text_to_sql_agent.llm import LLMResponse
from text_to_sql_agent.run_log import RunRecord
from text_to_sql_agent.tracing import Tracer

EXPORTER = InMemorySpanExporter()


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)

    def complete(self, system, user):
        return LLMResponse(text=self.responses.pop(0), model="fake", prompt_tokens=10, completion_tokens=5, latency_ms=1.0)


@pytest.fixture(scope="module")
def client():
    return Langfuse(public_key="pk-lf-test", secret_key="sk-lf-test", base_url="http://127.0.0.1:9", span_exporter=EXPORTER)


@pytest.fixture
def tracer(client):
    EXPORTER.clear()
    return Tracer(client)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "people.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE people (id INTEGER, name TEXT, age INTEGER)")
    con.executemany("INSERT INTO people VALUES (?, ?, ?)", [(i, f"p{i}", 20 + i) for i in range(1, 8)])
    con.commit()
    con.close()
    return path


def exported_spans():
    return sorted(EXPORTER.get_finished_spans(), key=lambda span: span.start_time)


def attr(span, key):
    value = span.attributes[key]
    return json.loads(value) if isinstance(value, str) and value[:1] in "[{" else value


def test_first_attempt_success_produces_one_trace_with_a_span_per_node(db, tracer):
    run = run_agent("Who is over 25?", db, ScriptedLLM("SELECT name FROM people WHERE age > 25 ORDER BY id"), tracer=tracer)
    spans = exported_spans()
    assert [span.name for span in spans] == ["text_to_sql", "generate", "guardrail", "execute"]
    root, generate, guardrail, execute = spans
    assert root.parent is None
    assert [span.parent.span_id for span in (generate, guardrail, execute)] == [root.context.span_id] * 3
    assert run.trace_id == format(root.context.trace_id, "032x")
    assert RunRecord.from_run(run).trace_id == run.trace_id
    assert attr(root, "langfuse.observation.type") == "agent"
    assert attr(root, "langfuse.observation.input") == {"question": "Who is over 25?"}
    assert attr(root, "langfuse.observation.output") == {
        "status": "success",
        "final_sql": "SELECT name FROM people WHERE age > 25 ORDER BY id",
        "attempts": 1,
    }
    assert attr(generate, "langfuse.observation.type") == "generation"
    assert attr(generate, "langfuse.observation.model.name") == "fake"
    assert attr(generate, "langfuse.observation.usage_details") == {"input": 10, "output": 5, "total": 15}
    assert attr(generate, "langfuse.observation.output") == "SELECT name FROM people WHERE age > 25 ORDER BY id"
    assert attr(generate, "langfuse.observation.input")[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert attr(guardrail, "langfuse.observation.type") == "guardrail"
    assert attr(guardrail, "langfuse.observation.level") == "DEFAULT"
    assert attr(guardrail, "langfuse.observation.output") == {"decision": "allow", "reason_code": None, "reason": None}
    assert attr(execute, "langfuse.observation.type") == "tool"
    assert attr(execute, "langfuse.observation.output") == {
        "columns": ["name"],
        "row_count": 2,
        "rows_preview": [["p6"], ["p7"]],
        "truncated": False,
    }
    assert [attr(span, "langfuse.observation.metadata.attempt") for span in (generate, guardrail, execute)] == [1, 1, 1]


def test_retries_appear_as_flagged_steps_in_order(db, tracer):
    llm = ScriptedLLM("DELETE FROM people", "SELECT nme FROM people", "SELECT COUNT(*) FROM people")
    run = run_agent("How many?", db, llm, tracer=tracer)
    assert run.status == "success"
    spans = exported_spans()
    assert [span.name for span in spans] == [
        "text_to_sql",
        "generate",
        "guardrail",
        "observe",
        "generate",
        "guardrail",
        "execute",
        "observe",
        "generate",
        "guardrail",
        "execute",
    ]
    assert [attr(span, "langfuse.observation.metadata.attempt") for span in spans[1:]] == [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
    rejected = spans[2]
    assert attr(rejected, "langfuse.observation.level") == "WARNING"
    assert attr(rejected, "langfuse.observation.status_message") == "Rejected write statement (DELETE); only read-only SELECT is allowed."
    assert attr(spans[3], "langfuse.observation.output")["stage"] == "guardrail"
    assert attr(spans[3], "langfuse.observation.output")["exhausted"] is False
    failed = spans[6]
    assert attr(failed, "langfuse.observation.level") == "ERROR"
    assert attr(failed, "langfuse.observation.status_message") == "OperationalError: no such column: nme"
    assert attr(failed, "langfuse.observation.output") == {"error": "OperationalError: no such column: nme"}
    assert attr(spans[7], "langfuse.observation.output")["stage"] == "execution"
    assert attr(spans[0], "langfuse.observation.output") == {"status": "success", "final_sql": "SELECT COUNT(*) FROM people", "attempts": 3}


def test_exhausted_run_marks_last_observe_step(db, tracer):
    run = run_agent("Anything?", db, ScriptedLLM("SELECT x FROM nowhere", "SELECT y FROM nowhere"), max_attempts=2, tracer=tracer)
    assert run.status == "failed"
    spans = exported_spans()
    assert [span.name for span in spans[-2:]] == ["execute", "observe"]
    assert attr(spans[-1], "langfuse.observation.output")["exhausted"] is True
    assert attr(spans[0], "langfuse.observation.output") == {"status": "failed", "final_sql": None, "attempts": 2}


def test_tracer_is_disabled_without_keys(db, monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr("text_to_sql_agent.tracing.load_dotenv", lambda: None)
    tracer = Tracer()
    assert tracer.enabled is False
    run = run_agent("One?", db, ScriptedLLM("SELECT 1"), tracer=tracer)
    assert run.status == "success"
    assert run.trace_id is None
    assert RunRecord.from_run(run).trace_id is None

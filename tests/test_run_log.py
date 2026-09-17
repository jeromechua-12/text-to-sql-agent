import logging
import sqlite3

import pytest

from text_to_sql_agent.agent import run_agent
from text_to_sql_agent.guardrail import GuardrailDecision, RejectionReason
from text_to_sql_agent.llm import LLMResponse
from text_to_sql_agent.run_log import SCHEMA_VERSION, RunLogger, RunRecord, load_records


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)

    def complete(self, system, user):
        return LLMResponse(text=self.responses.pop(0), model="fake", prompt_tokens=10, completion_tokens=5, latency_ms=1.0)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "people.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE people (id INTEGER, name TEXT, age INTEGER)")
    con.executemany("INSERT INTO people VALUES (?, ?, ?)", [(i, f"p{i}", 20 + i) for i in range(1, 8)])
    con.commit()
    con.close()
    return path


def test_record_for_first_attempt_success(db):
    run = run_agent("Who is over 25?", db, ScriptedLLM("SELECT name FROM people WHERE age > 25 ORDER BY id"))
    record = RunRecord.from_run(run, run_id="run-1")
    assert record.schema_version == SCHEMA_VERSION
    assert record.run_id == "run-1"
    assert record.status == "success"
    assert record.final_sql == "SELECT name FROM people WHERE age > 25 ORDER BY id"
    assert record.attempt_count == 1
    assert record.retry_count == 0
    assert record.recovered is False
    assert record.model == "fake"
    assert record.prompt_tokens == 10
    assert record.completion_tokens == 5
    assert record.cost_usd == 0.0
    assert record.question == "Who is over 25?"
    assert record.schema_text == "CREATE TABLE people (id INTEGER, name TEXT, age INTEGER)"
    attempt = record.attempts[0]
    assert attempt.outcome == "success"
    assert attempt.guardrail.decision is GuardrailDecision.ALLOW
    assert attempt.execution.ok is True
    assert attempt.execution.columns == ["name"]
    assert attempt.execution.row_count == 2
    assert attempt.execution.rows_preview == [["p6"], ["p7"]]
    assert attempt.observation is None


def test_record_marks_recovery_after_guardrail_rejection(db):
    run = run_agent("How many?", db, ScriptedLLM("DELETE FROM people", "SELECT COUNT(*) FROM people"))
    record = RunRecord.from_run(run)
    assert record.status == "success"
    assert record.attempt_count == 2
    assert record.retry_count == 1
    assert record.recovered is True
    assert record.prompt_tokens == 20
    first = record.attempts[0]
    assert first.outcome == "guardrail_rejected"
    assert first.guardrail.decision is GuardrailDecision.REJECT
    assert first.guardrail.reason_code is RejectionReason.FORBIDDEN_STATEMENT
    assert first.execution is None
    assert first.observation.stage == "guardrail"
    assert record.attempts[1].outcome == "success"
    assert record.attempts[1].execution.rows_preview == [[7]]


def test_record_for_exhausted_run(db):
    run = run_agent("Anything?", db, ScriptedLLM("SELECT x FROM nowhere", "SELECT y FROM nowhere"), max_attempts=2)
    record = RunRecord.from_run(run)
    assert record.status == "failed"
    assert record.final_sql is None
    assert record.recovered is False
    assert record.retry_count == 1
    last = record.attempts[-1]
    assert last.outcome == "execution_error"
    assert last.execution.ok is False
    assert last.execution.error == "OperationalError: no such table: nowhere"
    assert last.observation.stage == "execution"
    assert last.observation.error == "OperationalError: no such table: nowhere"


def test_rows_preview_is_capped(db):
    run = run_agent("Everyone?", db, ScriptedLLM("SELECT id FROM people ORDER BY id"))
    execution = RunRecord.from_run(run).attempts[0].execution
    assert execution.row_count == 7
    assert execution.rows_preview == [[1], [2], [3], [4], [5]]
    assert execution.truncated is False


def test_logger_appends_jsonl_and_round_trips(db, tmp_path, caplog):
    log_path = tmp_path / "logs" / "runs.jsonl"
    run_logger = RunLogger(log_path)
    with caplog.at_level(logging.INFO, logger="text_to_sql_agent.run_log"):
        first = run_logger.log(run_agent("One?", db, ScriptedLLM("SELECT 1")), spider_id="dev_0")
        second = run_logger.log(run_agent("Two?", db, ScriptedLLM("SELECT 2")), trace_id="trace-abc")
    assert log_path.read_text().count("\n") == 2
    loaded = load_records(log_path)
    assert loaded == [first, second]
    assert loaded[0].metadata == {"spider_id": "dev_0"}
    assert loaded[0].trace_id is None
    assert loaded[1].trace_id == "trace-abc"
    assert loaded[1].attempts[0].execution.rows_preview == [[2]]
    assert first.run_id != second.run_id
    assert f"run {first.run_id} success attempts=1 retries=0 recovered=False tokens=15" in caplog.text

import sqlite3

import pytest

from text_to_sql_agent.agent import Attempt, extract_sql, package_observation, run_agent
from text_to_sql_agent.guardrail import RejectionReason
from text_to_sql_agent.llm import LLMResponse
from text_to_sql_agent.sandbox import execute_query, schema_ddl
from text_to_sql_agent.schema_retrieval import SchemaRetriever


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        return LLMResponse(text=self.responses.pop(0), model="fake", prompt_tokens=10, completion_tokens=5, latency_ms=1.0)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "people.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE people (id INTEGER, name TEXT, age INTEGER)")
    con.executemany("INSERT INTO people VALUES (?, ?, ?)", [(1, "ada", 36), (2, "bob", 25), (3, "cy", 41)])
    con.commit()
    con.close()
    return path


def test_first_attempt_success(db):
    llm = ScriptedLLM("SELECT name FROM people WHERE age > 30 ORDER BY name")
    run = run_agent("Who is over 30?", db, llm)
    assert run.status == "success"
    assert len(run.attempts) == 1
    assert run.final_sql == "SELECT name FROM people WHERE age > 30 ORDER BY name"
    assert run.result.rows == [("ada",), ("cy",)]
    assert run.attempts[0].verdict.allowed
    assert run.attempts[0].observation is None
    assert run.prompt_tokens == 10
    assert "CREATE TABLE people" in llm.prompts[0]
    assert run.retrieval is None


def test_retriever_prunes_the_schema_shown_to_the_model(concert_db, concert_ddl, embedder):
    llm = ScriptedLLM("SELECT name FROM singer WHERE country = 'France'")
    run = run_agent("Which singers come from France?", concert_db, llm, retriever=SchemaRetriever(embedder, top_k=1))
    assert run.status == "success"
    assert run.result.rows == [("Joe",)]
    assert run.schema_text == concert_ddl["singer"]
    assert [match.name for match in run.retrieval.tables] == ["singer"]
    assert run.retrieval.total_tables == 4
    assert f"Schema:\n{concert_ddl['singer']}\n\nQuestion:" in llm.prompts[0]
    assert "CREATE TABLE stadium" not in llm.prompts[0]


def test_guardrail_rejection_feeds_back_and_recovers(db):
    llm = ScriptedLLM("DELETE FROM people", "SELECT COUNT(*) FROM people")
    run = run_agent("How many people?", db, llm)
    first = run.attempts[0]
    assert first.verdict.reason_code is RejectionReason.FORBIDDEN_STATEMENT
    assert first.execution is None
    assert first.observation.stage == "guardrail"
    assert first.observation.previous_sql == "DELETE FROM people"
    assert "DELETE FROM people" in llm.prompts[1]
    assert first.verdict.reason in llm.prompts[1]
    assert run.status == "success"
    assert run.result.rows == [(3,)]


def test_execution_error_feeds_back_and_recovers(db):
    llm = ScriptedLLM("SELECT nme FROM people", "SELECT name FROM people WHERE id = 1")
    run = run_agent("Name of person 1?", db, llm)
    first = run.attempts[0]
    assert first.verdict.allowed
    assert first.execution.error == "OperationalError: no such column: nme"
    assert first.observation.stage == "execution"
    assert first.observation.hint.startswith("That column does not exist")
    assert "no such column: nme" in llm.prompts[1]
    assert run.status == "success"
    assert run.result.rows == [("ada",)]


def test_gives_up_after_max_attempts(db):
    llm = ScriptedLLM("SELECT x FROM nowhere", "SELECT y FROM nowhere", "SELECT z FROM nowhere")
    run = run_agent("Anything?", db, llm, max_attempts=3)
    assert run.status == "failed"
    assert len(run.attempts) == 3
    assert len(llm.prompts) == 3
    assert run.final_sql is None
    assert run.result is None
    assert all(attempt.observation is not None for attempt in run.attempts)
    assert run.attempts[2].observation.attempt == 3


def test_prompt_accumulates_all_prior_observations(db):
    llm = ScriptedLLM("SELECT x FROM nowhere", "DROP TABLE people", "SELECT 1")
    run = run_agent("Anything?", db, llm)
    assert run.status == "success"
    assert "Attempt 1 failed at the execution stage." in llm.prompts[2]
    assert "Attempt 2 failed at the guardrail stage." in llm.prompts[2]


def test_package_observation_rejects_healthy_attempt():
    response = LLMResponse(text="SELECT 1", model="fake", prompt_tokens=1, completion_tokens=1, latency_ms=1.0)
    attempt = Attempt(index=1, raw_response="SELECT 1", sql="SELECT 1", llm=response)
    with pytest.raises(ValueError, match="did not fail"):
        package_observation(attempt)


def test_extract_sql_strips_fence():
    assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("  SELECT 2  ") == "SELECT 2"


def test_execute_query_truncates_rows(db):
    result = execute_query(db, "SELECT id FROM people ORDER BY id", row_limit=2)
    assert result.ok
    assert result.columns == ["id"]
    assert result.rows == [(1,), (2,)]
    assert result.truncated


def test_execute_query_captures_error(db):
    result = execute_query(db, "SELECT * FROM missing")
    assert not result.ok
    assert result.error == "OperationalError: no such table: missing"
    assert result.rows == []


def test_schema_ddl_lists_tables(db):
    assert schema_ddl(db) == "CREATE TABLE people (id INTEGER, name TEXT, age INTEGER)"

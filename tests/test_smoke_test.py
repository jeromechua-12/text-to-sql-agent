import json
import sqlite3
from pathlib import Path

import pytest

from text_to_sql_agent.llm import LLMResponse
from text_to_sql_agent.run_log import RunLogger, load_records
from text_to_sql_agent.smoke_test import (
    ExampleResult,
    SmokeReport,
    db_path_for,
    execution_match,
    load_examples,
    main,
    percentile,
    run_smoke_test,
    summarise,
)
from text_to_sql_agent.tracing import Tracer


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)

    def complete(self, system, user):
        return LLMResponse(text=self.responses.pop(0), model="fake", prompt_tokens=10, completion_tokens=5, latency_ms=1.0)


@pytest.fixture
def db_root(tmp_path):
    root = tmp_path / "database"
    (root / "people").mkdir(parents=True)
    con = sqlite3.connect(root / "people" / "people.sqlite")
    con.execute("CREATE TABLE people (id INTEGER, name TEXT, age INTEGER)")
    con.executemany("INSERT INTO people VALUES (?, ?, ?)", [(1, "ada", 36), (2, "bob", 25), (3, "cy", 41)])
    con.commit()
    con.close()
    return root


@pytest.fixture
def people_db(db_root):
    return db_root / "people" / "people.sqlite"


@pytest.fixture
def tracer(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr("text_to_sql_agent.tracing.load_dotenv", lambda: None)
    tracer = Tracer()
    assert tracer.enabled is False
    return tracer


def write_dev_json(path, entries):
    path.write_text(json.dumps([{"db_id": db_id, "question": question, "query": query} for db_id, question, query in entries]))
    return path


def result(**overrides):
    fields = {
        "index": 0,
        "db_id": "people",
        "question": "q",
        "gold_sql": "SELECT 1",
        "predicted_sql": "SELECT 1",
        "status": "success",
        "correct": True,
        "attempt_count": 1,
        "first_attempt_failed": False,
        "recovered": False,
        "latency_ms": 100.0,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost_usd": 0.001,
        "run_id": "run",
    }
    return ExampleResult(**{**fields, **overrides})


def test_load_examples_keeps_every_kth_question(tmp_path):
    dev = write_dev_json(tmp_path / "dev.json", [("db", f"q{i}", f"SELECT {i}") for i in range(7)])
    examples = load_examples(dev, 3)
    assert [example.index for example in examples] == [0, 2, 4]
    assert examples[1].db_id == "db"
    assert examples[1].question == "q2"
    assert examples[1].gold_sql == "SELECT 2"


def test_load_examples_returns_everything_when_limit_covers_the_file(tmp_path):
    dev = write_dev_json(tmp_path / "dev.json", [("db", f"q{i}", f"SELECT {i}") for i in range(4)])
    assert [example.index for example in load_examples(dev, 10)] == [0, 1, 2, 3]


def test_db_path_for():
    assert db_path_for("data/spider_data/database", "concert_singer") == Path(
        "data/spider_data/database/concert_singer/concert_singer.sqlite"
    )


def test_execution_match_ignores_row_order_without_order_by(people_db):
    assert execution_match(people_db, "SELECT id FROM people ORDER BY id DESC", "SELECT id FROM people") is True


def test_execution_match_requires_row_order_with_order_by(people_db):
    gold = "SELECT name FROM people ORDER BY age"
    assert execution_match(people_db, "SELECT name FROM people ORDER BY age", gold) is True
    assert execution_match(people_db, "SELECT name FROM people ORDER BY age DESC", gold) is False


def test_execution_match_uses_multiset_semantics(people_db):
    assert execution_match(people_db, "SELECT DISTINCT age > 30 FROM people", "SELECT age > 30 FROM people") is False


def test_execution_match_rejects_different_columns(people_db):
    assert execution_match(people_db, "SELECT id, name FROM people", "SELECT id FROM people") is False


def test_execution_match_is_false_when_prediction_errors(people_db):
    assert execution_match(people_db, "SELECT nme FROM people", "SELECT name FROM people") is False


def test_execution_match_raises_when_gold_errors(people_db):
    with pytest.raises(ValueError, match="Gold SQL failed"):
        execution_match(people_db, "SELECT 1", "SELECT x FROM nowhere")


def test_percentile_uses_nearest_rank():
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.95) == 40.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.5) == 20.0
    assert percentile([5.0], 0.95) == 5.0


def test_summarise_aggregates_results():
    results = [
        result(correct=True, latency_ms=100.0),
        result(correct=True, first_attempt_failed=True, recovered=True, attempt_count=2, latency_ms=200.0),
        result(correct=False, first_attempt_failed=True, status="failed", predicted_sql=None, latency_ms=300.0),
        result(correct=False, latency_ms=400.0),
    ]
    summary = summarise(results)
    assert summary.examples == 4
    assert summary.correct == 2
    assert summary.execution_accuracy == 0.5
    assert summary.first_attempt_failures == 2
    assert summary.recovered == 1
    assert summary.recovery_rate == 0.5
    assert summary.latency_median_ms == 250.0
    assert summary.latency_p95_ms == 400.0
    assert summary.prompt_tokens == 40
    assert summary.completion_tokens == 20
    assert summary.cost_usd == pytest.approx(0.004)


def test_summarise_has_no_recovery_rate_without_first_attempt_failures():
    assert summarise([result(), result()]).recovery_rate is None


def test_run_smoke_test_scores_and_logs_every_example(db_root, people_db, tmp_path, tracer):
    dev = write_dev_json(
        tmp_path / "dev.json",
        [
            ("people", "How many people?", "SELECT count(*) FROM people"),
            ("people", "All names?", "SELECT name FROM people"),
            ("people", "Name of person 1?", "SELECT name FROM people WHERE id = 1"),
            ("people", "Anything?", "SELECT 1"),
        ],
    )
    llm = ScriptedLLM(
        "SELECT COUNT(*) FROM people",
        "SELECT nme FROM people",
        "SELECT name FROM people",
        "SELECT name FROM people WHERE id = 2",
        "DELETE FROM people",
        "SELECT x FROM nowhere",
    )
    log_path = tmp_path / "runs.jsonl"
    results = run_smoke_test(load_examples(dev, 4), db_root, llm, RunLogger(log_path), tracer, max_attempts=2, batch_id="batch-1")

    assert [r.status for r in results] == ["success", "success", "success", "failed"]
    assert [r.correct for r in results] == [True, True, False, False]
    assert [r.attempt_count for r in results] == [1, 2, 1, 2]
    assert [r.first_attempt_failed for r in results] == [False, True, False, True]
    assert [r.recovered for r in results] == [False, True, False, False]
    assert results[1].predicted_sql == "SELECT name FROM people"
    assert results[3].predicted_sql is None
    assert all(r.trace_id is None for r in results)

    records = load_records(log_path)
    assert [record.run_id for record in records] == [r.run_id for r in results]
    assert records[1].metadata == {
        "batch_id": "batch-1",
        "spider_index": 1,
        "db_id": "people",
        "gold_sql": "SELECT name FROM people",
        "correct": True,
    }
    assert records[3].metadata["correct"] is False
    assert records[3].attempts[0].outcome == "guardrail_rejected"

    summary = summarise(results)
    assert summary.examples == 4
    assert summary.correct == 2
    assert summary.execution_accuracy == 0.5
    assert summary.first_attempt_failures == 2
    assert summary.recovered == 1
    assert summary.recovery_rate == 0.5
    assert summary.prompt_tokens == 60
    assert summary.completion_tokens == 30


def test_main_writes_report_and_run_log(db_root, tmp_path, tracer, embedder, monkeypatch, capsys):
    dev = write_dev_json(
        tmp_path / "dev.json",
        [("people", "How many people?", "SELECT count(*) FROM people"), ("people", "Names by id?", "SELECT name FROM people ORDER BY id")],
    )
    monkeypatch.setattr(
        "text_to_sql_agent.smoke_test.OpenAIChat",
        lambda model: ScriptedLLM("SELECT COUNT(*) FROM people", "SELECT name FROM people ORDER BY id"),
    )
    monkeypatch.setattr("text_to_sql_agent.smoke_test.SentenceTransformerEmbedder", lambda model: embedder)
    output_dir = tmp_path / "out"
    main(["--n", "2", "--dev-json", str(dev), "--db-root", str(db_root), "--output-dir", str(output_dir), "--model", "fake-model"])

    batch_dirs = list(output_dir.iterdir())
    assert len(batch_dirs) == 1
    report = SmokeReport.model_validate_json((batch_dirs[0] / "report.json").read_text())
    assert report.batch_id == batch_dirs[0].name
    assert report.model == "fake-model"
    assert report.max_attempts == 3
    assert report.embedding_model == "all-MiniLM-L6-v2"
    assert report.top_k == 5
    assert report.summary.examples == 2
    assert report.summary.correct == 2
    assert report.summary.execution_accuracy == 1.0
    assert report.summary.recovery_rate is None
    assert [r.index for r in report.results] == [0, 1]

    records = load_records(batch_dirs[0] / "runs.jsonl")
    assert len(records) == 2
    assert {record.metadata["batch_id"] for record in records} == {report.batch_id}
    assert [match.name for match in records[0].retrieval.tables] == ["people"]

    out = capsys.readouterr().out
    assert "max 3 attempts, top 5 tables via all-MiniLM-L6-v2" in out
    assert "Execution accuracy: 2/2 = 100.0%" in out
    assert "Self-correction recovery: no first-attempt failures" in out
    assert f"Report written to {batch_dirs[0] / 'report.json'}" in out


def test_main_full_schema_flag_skips_retrieval(db_root, tmp_path, tracer, monkeypatch, capsys):
    dev = write_dev_json(tmp_path / "dev.json", [("people", "How many people?", "SELECT count(*) FROM people")])
    monkeypatch.setattr("text_to_sql_agent.smoke_test.OpenAIChat", lambda model: ScriptedLLM("SELECT COUNT(*) FROM people"))
    output_dir = tmp_path / "out"
    main(["--n", "1", "--dev-json", str(dev), "--db-root", str(db_root), "--output-dir", str(output_dir), "--full-schema"])

    batch_dir = next(output_dir.iterdir())
    report = SmokeReport.model_validate_json((batch_dir / "report.json").read_text())
    assert report.embedding_model is None
    assert report.top_k is None
    assert load_records(batch_dir / "runs.jsonl")[0].retrieval is None
    assert "max 3 attempts, full schema" in capsys.readouterr().out

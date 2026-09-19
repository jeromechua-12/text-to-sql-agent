import shutil

import pytest
from fastapi.testclient import TestClient

from text_to_sql_agent.api import Dependencies, create_app
from text_to_sql_agent.llm import LLMResponse
from text_to_sql_agent.run_log import RunLogger
from text_to_sql_agent.schema_retrieval import SchemaRetriever
from text_to_sql_agent.tracing import Tracer


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        return LLMResponse(text=self.responses.pop(0), model="fake", prompt_tokens=10, completion_tokens=5, latency_ms=1.0)


@pytest.fixture
def disabled_tracer(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr("text_to_sql_agent.tracing.load_dotenv", lambda: None)
    return Tracer()


@pytest.fixture
def db_root(tmp_path, concert_db):
    root = tmp_path / "database"
    (root / "concert").mkdir(parents=True)
    shutil.copy(concert_db, root / "concert" / "concert.sqlite")
    return root


def make_client(db_root, tmp_path, tracer, llm, retriever=None):
    deps = Dependencies(
        llm=llm,
        db_root=db_root,
        run_logger=RunLogger(tmp_path / "runs.jsonl"),
        tracer=tracer,
        retriever=retriever,
    )
    return TestClient(create_app(deps))


def test_health_reports_retrieval_state(db_root, tmp_path, disabled_tracer):
    with make_client(db_root, tmp_path, disabled_tracer, ScriptedLLM()) as client:
        assert client.get("/health").json() == {"status": "ok", "retrieval": False}


def test_databases_lists_available_ids(db_root, tmp_path, disabled_tracer):
    with make_client(db_root, tmp_path, disabled_tracer, ScriptedLLM()) as client:
        assert client.get("/databases").json() == {"databases": ["concert"]}


def test_ask_returns_result_and_final_sql(db_root, tmp_path, disabled_tracer):
    llm = ScriptedLLM("SELECT name FROM singer WHERE country = 'France'")
    with make_client(db_root, tmp_path, disabled_tracer, llm) as client:
        response = client.post("/ask", json={"question": "Which singers come from France?", "db_id": "concert"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "success"
    assert body["final_sql"] == "SELECT name FROM singer WHERE country = 'France'"
    assert body["result"] == {"columns": ["name"], "rows": [["Joe"]], "row_count": 1, "truncated": False}
    assert body["retry_count"] == 0
    assert body["prompt_tokens"] == 10


def test_ask_recovers_and_reports_the_retry(db_root, tmp_path, disabled_tracer):
    llm = ScriptedLLM("SELECT nme FROM singer", "SELECT name FROM singer WHERE country = 'France'")
    with make_client(db_root, tmp_path, disabled_tracer, llm) as client:
        body = client.post("/ask", json={"question": "France singers?", "db_id": "concert"}).json()
    assert body["status"] == "success"
    assert body["retry_count"] == 1
    assert body["recovered"] is True
    assert body["attempts"][0]["outcome"] == "execution_error"
    assert body["result"]["rows"] == [["Joe"]]


def test_ask_uses_the_retriever_when_configured(db_root, tmp_path, disabled_tracer, embedder):
    llm = ScriptedLLM("SELECT name FROM singer WHERE country = 'France'")
    retriever = SchemaRetriever(embedder, top_k=1)
    with make_client(db_root, tmp_path, disabled_tracer, llm, retriever=retriever) as client:
        body = client.post("/ask", json={"question": "Which singers come from France?", "db_id": "concert"}).json()
    assert body["status"] == "success"
    assert [table["name"] for table in body["retrieval"]["tables"]] == ["singer"]
    assert "CREATE TABLE stadium" not in llm.prompts[0]


def test_ask_unknown_database_is_404(db_root, tmp_path, disabled_tracer):
    with make_client(db_root, tmp_path, disabled_tracer, ScriptedLLM()) as client:
        response = client.post("/ask", json={"question": "anything?", "db_id": "missing"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown database 'missing'."


def test_ask_logs_one_record_per_run(db_root, tmp_path, disabled_tracer):
    llm = ScriptedLLM("SELECT name FROM singer WHERE country = 'France'")
    with make_client(db_root, tmp_path, disabled_tracer, llm) as client:
        client.post("/ask", json={"question": "France singers?", "db_id": "concert"})
    from text_to_sql_agent.run_log import load_records

    records = load_records(tmp_path / "runs.jsonl")
    assert len(records) == 1
    assert records[0].metadata == {"source": "api", "db_id": "concert"}

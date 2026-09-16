import sqlite3

import pytest

from text_to_sql_agent.guardrail import GuardrailDecision, RejectionReason, check_query
from text_to_sql_agent.sandbox import read_only_connection

ALLOWED = [
    "SELECT 1",
    "SELECT name, age FROM people WHERE age > 30",
    "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
    "SELECT a FROM t UNION SELECT b FROM u",
    "SELECT department, COUNT(*) FROM staff GROUP BY department ORDER BY 2 DESC",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allows_read_only_selects(sql):
    verdict = check_query(sql)
    assert verdict.decision is GuardrailDecision.ALLOW
    assert verdict.allowed
    assert verdict.reason_code is None


REJECTED = [
    ("INSERT INTO t VALUES (1)", RejectionReason.FORBIDDEN_STATEMENT),
    ("UPDATE t SET a = 1", RejectionReason.FORBIDDEN_STATEMENT),
    ("DELETE FROM t", RejectionReason.FORBIDDEN_STATEMENT),
    ("DROP TABLE t", RejectionReason.FORBIDDEN_STATEMENT),
    ("CREATE TABLE t (a INT)", RejectionReason.FORBIDDEN_STATEMENT),
    ("ALTER TABLE t ADD COLUMN b INT", RejectionReason.FORBIDDEN_STATEMENT),
    ("PRAGMA table_info(t)", RejectionReason.FORBIDDEN_STATEMENT),
    ("ATTACH DATABASE 'x.db' AS y", RejectionReason.FORBIDDEN_STATEMENT),
    ("SELECT 1; SELECT 2", RejectionReason.MULTIPLE_STATEMENTS),
    ("SELECT 1; DROP TABLE t", RejectionReason.MULTIPLE_STATEMENTS),
    ("", RejectionReason.EMPTY_QUERY),
    ("   ", RejectionReason.EMPTY_QUERY),
]


@pytest.mark.parametrize("sql, code", REJECTED)
def test_rejects_unsafe_queries(sql, code):
    verdict = check_query(sql)
    assert verdict.decision is GuardrailDecision.REJECT
    assert not verdict.allowed
    assert verdict.reason_code is code
    assert verdict.reason


def test_rejection_carries_reason_for_feedback():
    verdict = check_query("DELETE FROM t")
    assert verdict.reason and "DELETE" in verdict.reason
    assert verdict.sql == "DELETE FROM t"


def test_garbage_input_is_rejected():
    verdict = check_query("SELCT bad frm nowhere")
    assert verdict.decision is GuardrailDecision.REJECT
    assert verdict.reason_code is RejectionReason.PARSE_ERROR


def _seed_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE people (id INTEGER, name TEXT)")
    con.execute("INSERT INTO people VALUES (1, 'ada')")
    con.commit()
    con.close()


def test_read_only_connection_allows_select(tmp_path):
    db = tmp_path / "test.sqlite"
    _seed_db(db)
    con = read_only_connection(db)
    assert con.execute("SELECT name FROM people").fetchone() == ("ada",)
    con.close()


def test_read_only_connection_blocks_writes(tmp_path):
    db = tmp_path / "test.sqlite"
    _seed_db(db)
    con = read_only_connection(db)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("INSERT INTO people VALUES (2, 'bob')")
    con.close()

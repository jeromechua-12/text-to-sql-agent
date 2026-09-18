"""Read-only SQLite sandbox: opens execution connections that cannot mutate the database."""

import sqlite3
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ExecutionResult(BaseModel):
    """Outcome of running one query in the sandbox: rows on success, the raw database error otherwise."""

    columns: list[str] = []
    rows: list[tuple[Any, ...]] = []
    truncated: bool = False
    error: str | None = None
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


def read_only_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection in read-only mode with query_only enforced as defense in depth."""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def execute_query(db_path: str | Path, sql: str, row_limit: int | None = 100) -> ExecutionResult:
    """Run sql over a read-only connection, returning rows or the raw sqlite error as data."""
    started = time.perf_counter()
    connection = read_only_connection(db_path)
    try:
        cursor = connection.execute(sql)
        rows = cursor.fetchall() if row_limit is None else cursor.fetchmany(row_limit + 1)
        columns = [column[0] for column in cursor.description or []]
    except sqlite3.Error as err:
        return ExecutionResult(error=f"{type(err).__name__}: {err}", latency_ms=_elapsed_ms(started))
    finally:
        connection.close()
    truncated = row_limit is not None and len(rows) > row_limit
    return ExecutionResult(
        columns=columns,
        rows=rows[:row_limit] if truncated else rows,
        truncated=truncated,
        latency_ms=_elapsed_ms(started),
    )


def schema_ddl(db_path: str | Path) -> str:
    """Return the CREATE TABLE statements of every user table, the full-schema fallback used when no retriever is configured."""
    connection = read_only_connection(db_path)
    rows = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    connection.close()
    return "\n\n".join(row[0] for row in rows)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000

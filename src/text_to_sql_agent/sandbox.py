"""Read-only SQLite sandbox: opens execution connections that cannot mutate the database."""

import sqlite3
from pathlib import Path


def read_only_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection in read-only mode with query_only enforced as defense in depth."""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection

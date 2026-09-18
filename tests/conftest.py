import re
import sqlite3

import numpy as np
import pytest

CONCERT_DDL = {
    "concert": "CREATE TABLE concert (concert_id INTEGER, concert_name TEXT, stadium_id INTEGER, year INTEGER)",
    "singer": "CREATE TABLE singer (singer_id INTEGER, name TEXT, country TEXT)",
    "singer_in_concert": "CREATE TABLE singer_in_concert (concert_id INTEGER, singer_id INTEGER)",
    "stadium": "CREATE TABLE stadium (stadium_id INTEGER, name TEXT, capacity INTEGER)",
}


class BagOfWordsEmbedder:
    """Deterministic test embedder: one dimension per distinct token, assigned on first sight, counts as weights."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.slots: dict[str, int] = {}
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                vectors[row, self.slots.setdefault(token, len(self.slots))] += 1
        return vectors


@pytest.fixture
def embedder():
    return BagOfWordsEmbedder()


@pytest.fixture
def concert_ddl():
    return CONCERT_DDL


@pytest.fixture
def concert_db(tmp_path):
    path = tmp_path / "concert.sqlite"
    con = sqlite3.connect(path)
    for ddl in CONCERT_DDL.values():
        con.execute(ddl)
    con.executemany("INSERT INTO stadium VALUES (?, ?, ?)", [(1, "Stark Stadium", 100)])
    con.executemany("INSERT INTO singer VALUES (?, ?, ?)", [(1, "Joe", "France"), (2, "Ann", "Netherlands")])
    con.executemany("INSERT INTO concert VALUES (?, ?, ?, ?)", [(1, "Summer Fest", 1, 2014)])
    con.executemany("INSERT INTO singer_in_concert VALUES (?, ?)", [(1, 1)])
    con.commit()
    con.close()
    return path

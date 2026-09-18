"""Schema retrieval: embed one description per table and per column, index them in FAISS, and keep only the tables a question needs."""

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import faiss
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from text_to_sql_agent.sandbox import read_only_connection

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
SAMPLE_VALUES = 3
SAMPLE_VALUE_MAX_CHARS = 30
TEXT_AFFINITY = ("CHAR", "CLOB", "TEXT")

CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


class Column(BaseModel):
    """One column of a table with a few sample values when it holds text."""

    name: str
    type: str
    samples: list[str] = []


class Table(BaseModel):
    """One user table: its original CREATE statement and its columns."""

    name: str
    ddl: str
    columns: list[Column]


class SchemaDocument(BaseModel):
    """One embeddable description of a table or a column, tagged with the table it pulls into the prompt."""

    table: str
    kind: Literal["table", "column"]
    label: str
    text: str


class TableMatch(BaseModel):
    """A table chosen for the prompt, with its best score and the document that earned it."""

    name: str
    score: float
    matched: str


class SchemaRetrieval(BaseModel):
    """Outcome of one retrieval: the ranked tables kept for the prompt and the schema text rendered from them."""

    total_tables: int
    top_k: int
    tables: list[TableMatch]
    schema_text: str


class Embedder(Protocol):
    """Anything that turns a list of texts into one embedding row per text."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model, loaded once at construction."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed texts into unit-length vectors."""
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


@dataclass
class SchemaIndex:
    """One database's tables, their documents, and the FAISS index over those documents, built once and reused."""

    tables: list[Table]
    documents: list[SchemaDocument]
    index: faiss.Index


def humanise(identifier: str) -> str:
    """Turn an identifier such as StuID or singer_in_concert into lower-case words for embedding."""
    spaced = CAMEL_BOUNDARY.sub(" ", identifier.replace("_", " ").replace("-", " "))
    return " ".join(spaced.lower().split())


def quote(identifier: str) -> str:
    """Double-quote an identifier for use in introspection queries."""
    return '"' + identifier.replace('"', '""') + '"'


def load_catalog(db_path: str | Path) -> list[Table]:
    """Read every user table, its columns, and sample values for text columns over a read-only connection."""
    connection = read_only_connection(db_path)
    rows = connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    tables = []
    for name, ddl in rows:
        columns = []
        for info in connection.execute(f"PRAGMA table_info({quote(name)})").fetchall():
            column_name, column_type = info[1], info[2]
            samples = sample_values(connection, name, column_name) if has_text_affinity(column_type) else []
            columns.append(Column(name=column_name, type=column_type, samples=samples))
        tables.append(Table(name=name, ddl=ddl, columns=columns))
    connection.close()
    return tables


def has_text_affinity(column_type: str) -> bool:
    """Apply SQLite's rule for text affinity: the declared type mentions CHAR, CLOB, or TEXT."""
    upper = column_type.upper()
    return any(marker in upper for marker in TEXT_AFFINITY)


def sample_values(connection: sqlite3.Connection, table: str, column: str) -> list[str]:
    """Fetch a few distinct non-null values of a column, shortened so they stay embeddable."""
    rows = connection.execute(
        f"SELECT DISTINCT {quote(column)} FROM {quote(table)} WHERE {quote(column)} IS NOT NULL LIMIT {SAMPLE_VALUES}"
    ).fetchall()
    return [" ".join(str(row[0]).split())[:SAMPLE_VALUE_MAX_CHARS] for row in rows]


def describe(table: Table) -> list[SchemaDocument]:
    """Serialise a table into one document for the table as a whole and one per column."""
    column_words = ", ".join(humanise(column.name) for column in table.columns)
    documents = [
        SchemaDocument(
            table=table.name,
            kind="table",
            label=table.name,
            text=f"table {humanise(table.name)} with columns {column_words}",
        )
    ]
    for column in table.columns:
        text = f"{humanise(column.name)} of {humanise(table.name)}"
        if column.samples:
            text += ", for example " + ", ".join(column.samples)
        documents.append(SchemaDocument(table=table.name, kind="column", label=f"{table.name}.{column.name}", text=text))
    return documents


def build_index(tables: list[Table], embedder: Embedder) -> SchemaIndex:
    """Embed every document of the catalog and load the unit vectors into a flat inner-product FAISS index."""
    documents = [document for table in tables for document in describe(table)]
    vectors = np.ascontiguousarray(embedder.encode([document.text for document in documents]), dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return SchemaIndex(tables=tables, documents=documents, index=index)


class SchemaRetriever:
    """Indexes each database's schema once and returns the top-k tables whose descriptions best match a question."""

    def __init__(self, embedder: Embedder, top_k: int = DEFAULT_TOP_K) -> None:
        self.embedder = embedder
        self.top_k = top_k
        self.indexes: dict[str, SchemaIndex] = {}

    def index_for(self, db_path: str | Path) -> SchemaIndex:
        """Build or reuse the index for one database, keyed by its resolved path."""
        key = str(Path(db_path).resolve())
        if key not in self.indexes:
            self.indexes[key] = build_index(load_catalog(db_path), self.embedder)
        return self.indexes[key]

    def retrieve(self, db_path: str | Path, question: str, top_k: int | None = None) -> SchemaRetrieval:
        """Score every table by its best-matching document and render the DDL of the top-k tables, best first."""
        schema = self.index_for(db_path)
        k = top_k if top_k is not None else self.top_k
        query = np.ascontiguousarray(self.embedder.encode([question]), dtype="float32")
        faiss.normalize_L2(query)
        scores, ids = schema.index.search(query, schema.index.ntotal)
        matches: dict[str, TableMatch] = {}
        for score, doc_id in zip(scores[0].tolist(), ids[0].tolist()):
            document = schema.documents[doc_id]
            if document.table not in matches:
                matches[document.table] = TableMatch(name=document.table, score=round(score, 4), matched=document.label)
        kept = list(matches.values())[:k]
        ddl = {table.name: table.ddl for table in schema.tables}
        return SchemaRetrieval(
            total_tables=len(schema.tables),
            top_k=k,
            tables=kept,
            schema_text="\n\n".join(ddl[match.name] for match in kept),
        )

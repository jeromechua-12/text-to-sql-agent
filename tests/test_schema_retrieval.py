import pytest

from text_to_sql_agent.schema_retrieval import Column, SchemaRetriever, describe, humanise, load_catalog


def test_humanise_splits_underscores_and_camel_case():
    assert humanise("singer_in_concert") == "singer in concert"
    assert humanise("StuID") == "stu id"
    assert humanise("Singer_ID") == "singer id"
    assert humanise("concertName") == "concert name"


def test_load_catalog_reads_tables_columns_and_text_samples(concert_db, concert_ddl):
    tables = load_catalog(concert_db)
    assert [table.name for table in tables] == ["concert", "singer", "singer_in_concert", "stadium"]
    singer = tables[1]
    assert singer.ddl == concert_ddl["singer"]
    assert singer.columns == [
        Column(name="singer_id", type="INTEGER", samples=[]),
        Column(name="name", type="TEXT", samples=["Joe", "Ann"]),
        Column(name="country", type="TEXT", samples=["France", "Netherlands"]),
    ]
    assert tables[0].columns[3] == Column(name="year", type="INTEGER", samples=[])


def test_describe_serialises_one_document_per_table_and_column(concert_db):
    singer = load_catalog(concert_db)[1]
    documents = describe(singer)
    assert [(doc.kind, doc.label, doc.text) for doc in documents] == [
        ("table", "singer", "table singer with columns singer id, name, country"),
        ("column", "singer.singer_id", "singer id of singer"),
        ("column", "singer.name", "name of singer, for example Joe, Ann"),
        ("column", "singer.country", "country of singer, for example France, Netherlands"),
    ]
    assert all(doc.table == "singer" for doc in documents)


def test_retrieve_ranks_tables_by_their_best_document(concert_db, concert_ddl, embedder):
    retriever = SchemaRetriever(embedder, top_k=3)
    retrieval = retriever.retrieve(concert_db, "capacity of the stadium where singers from France performed")
    assert retrieval.total_tables == 4
    assert retrieval.top_k == 3
    assert [match.name for match in retrieval.tables] == ["stadium", "concert", "singer"]
    assert [match.matched for match in retrieval.tables] == ["stadium.capacity", "concert.stadium_id", "singer.country"]
    assert retrieval.tables[0].score == pytest.approx(0.5774, abs=1e-4)
    assert retrieval.schema_text == "\n\n".join([concert_ddl["stadium"], concert_ddl["concert"], concert_ddl["singer"]])


def test_retrieve_keeps_every_table_when_top_k_exceeds_the_catalog(concert_db, embedder):
    retrieval = SchemaRetriever(embedder, top_k=10).retrieve(concert_db, "anything")
    assert sorted(match.name for match in retrieval.tables) == ["concert", "singer", "singer_in_concert", "stadium"]
    assert retrieval.top_k == 10


def test_retrieve_honours_per_call_top_k(concert_db, embedder):
    retriever = SchemaRetriever(embedder, top_k=3)
    retrieval = retriever.retrieve(concert_db, "Which singers come from France?", top_k=1)
    assert [match.name for match in retrieval.tables] == ["singer"]
    assert retrieval.tables[0].matched == "singer.country"


def test_index_is_built_once_per_database(concert_db, embedder):
    retriever = SchemaRetriever(embedder)
    retriever.retrieve(concert_db, "first question")
    retriever.retrieve(concert_db, "second question")
    assert embedder.calls == 3
    assert list(retriever.indexes) == [str(concert_db.resolve())]
    assert retriever.indexes[str(concert_db.resolve())].index.ntotal == 16

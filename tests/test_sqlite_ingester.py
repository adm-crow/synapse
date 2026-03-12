import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from synapse_core.sqlite_ingester import ingest_sqlite


def create_test_db(tmp_path, table_name, columns_def, rows):
    """Create a temp SQLite database with one table and return its path."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(f"CREATE TABLE {table_name} ({', '.join(columns_def)})")
    for row in rows:
        placeholders = ", ".join("?" * len(row))
        conn.execute(f"INSERT INTO {table_name} VALUES ({placeholders})", row)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def mock_chroma():
    collection = MagicMock()
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client), \
         patch("synapse_core.pipeline.embedding_functions.SentenceTransformerEmbeddingFunction"):
        yield collection


# --- basic ingestion ---

def test_ingest_sqlite_basic(mock_chroma, tmp_path):
    db = create_test_db(
        tmp_path, "articles",
        ["id INTEGER PRIMARY KEY", "title TEXT", "body TEXT"],
        [(1, "Hello", "word " * 30), (2, "World", "text " * 30)],
    )
    ingest_sqlite(db_path=db, table="articles", chroma_path=str(tmp_path / "db"), verbose=False)
    assert mock_chroma.upsert.call_count == 2


def test_ingest_sqlite_is_idempotent(mock_chroma, tmp_path):
    db = create_test_db(
        tmp_path, "docs",
        ["id INTEGER PRIMARY KEY", "content TEXT"],
        [(1, "word " * 30)],
    )
    ingest_sqlite(db_path=db, table="docs", chroma_path=str(tmp_path / "db"), verbose=False)
    ingest_sqlite(db_path=db, table="docs", chroma_path=str(tmp_path / "db"), verbose=False)
    assert mock_chroma.upsert.call_count == 2


# --- column selection ---

def test_ingest_sqlite_with_columns(mock_chroma, tmp_path):
    db = create_test_db(
        tmp_path, "products",
        ["id INTEGER PRIMARY KEY", "name TEXT", "price REAL", "description TEXT"],
        [(1, "Widget", 9.99, "A great widget " * 10)],
    )
    ingest_sqlite(
        db_path=db, table="products",
        columns=["name", "description"],
        chroma_path=str(tmp_path / "db"), verbose=False,
    )
    doc = mock_chroma.upsert.call_args.kwargs["documents"][0]
    assert "Widget" in doc
    assert "9.99" not in doc  # price excluded


# --- row template ---

def test_ingest_sqlite_with_template(mock_chroma, tmp_path):
    db = create_test_db(
        tmp_path, "news",
        ["id INTEGER PRIMARY KEY", "title TEXT", "content TEXT"],
        [(1, "Big News", "Something important " * 10)],
    )
    ingest_sqlite(
        db_path=db, table="news",
        row_template="{title}: {content}",
        chroma_path=str(tmp_path / "db"), verbose=False,
    )
    doc = mock_chroma.upsert.call_args.kwargs["documents"][0]
    assert doc.startswith("Big News:")


# --- metadata ---

def test_ingest_sqlite_metadata_structure(mock_chroma, tmp_path):
    db = create_test_db(
        tmp_path, "docs",
        ["id INTEGER PRIMARY KEY", "content TEXT"],
        [(1, "word " * 30)],
    )
    ingest_sqlite(db_path=db, table="docs", chroma_path=str(tmp_path / "db"), verbose=False)
    meta = mock_chroma.upsert.call_args.kwargs["metadatas"][0]
    assert meta["source_type"] == "sqlite"
    assert "docs" in meta["source"]
    assert "row_id" in meta
    assert "chunk" in meta


# --- rowid fallback ---

def test_ingest_sqlite_uses_rowid_when_no_id_column(mock_chroma, tmp_path):
    db = create_test_db(
        tmp_path, "notes",
        ["title TEXT", "body TEXT"],  # no explicit id column
        [("Note A", "content " * 20), ("Note B", "content " * 20)],
    )
    ingest_sqlite(
        db_path=db, table="notes",
        id_column="id",  # doesn't exist — should fall back to rowid
        chroma_path=str(tmp_path / "db"), verbose=False,
    )
    assert mock_chroma.upsert.call_count == 2


# --- error cases ---

def test_ingest_sqlite_missing_db(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_sqlite(db_path=str(tmp_path / "missing.db"), table="foo",
                      chroma_path=str(tmp_path / "db"))


def test_ingest_sqlite_missing_table(tmp_path):
    db = create_test_db(tmp_path, "articles",
                        ["id INTEGER PRIMARY KEY", "title TEXT"], [(1, "Hello")])
    with pytest.raises(ValueError, match="Table"):
        ingest_sqlite(db_path=db, table="nonexistent", chroma_path=str(tmp_path / "db"))


def test_ingest_sqlite_invalid_columns(tmp_path):
    db = create_test_db(tmp_path, "articles",
                        ["id INTEGER PRIMARY KEY", "title TEXT"], [(1, "Hello")])
    with pytest.raises(ValueError, match="Columns"):
        ingest_sqlite(db_path=db, table="articles", columns=["nonexistent"],
                      chroma_path=str(tmp_path / "db"))


def test_ingest_sqlite_empty_table(mock_chroma, tmp_path):
    db = create_test_db(tmp_path, "empty",
                        ["id INTEGER PRIMARY KEY", "title TEXT"], [])
    ingest_sqlite(db_path=db, table="empty", chroma_path=str(tmp_path / "db"), verbose=False)
    mock_chroma.upsert.assert_not_called()


def test_ingest_sqlite_template_missing_column(mock_chroma, tmp_path):
    db = create_test_db(
        tmp_path, "articles",
        ["id INTEGER PRIMARY KEY", "title TEXT"],
        [(1, "Hello")],
    )
    with pytest.raises(ValueError, match="row_template"):
        ingest_sqlite(
            db_path=db, table="articles",
            row_template="{title}: {nonexistent}",
            chroma_path=str(tmp_path / "db"), verbose=False,
        )


def test_ingest_sqlite_template_null_value_renders_empty(mock_chroma, tmp_path):
    """NULL column values must render as '' in row_template, not 'None'."""
    db = create_test_db(
        tmp_path, "articles",
        ["id INTEGER PRIMARY KEY", "title TEXT", "body TEXT"],
        [(1, None, "body word " * 20)],
    )
    ingest_sqlite(
        db_path=db, table="articles",
        row_template="{title}: {body}",
        chroma_path=str(tmp_path / "db"), verbose=False,
    )
    docs = mock_chroma.upsert.call_args.kwargs["documents"]
    assert all("None" not in d for d in docs)

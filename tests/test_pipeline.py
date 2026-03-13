from unittest.mock import MagicMock, patch

import pytest

from synapse_core.pipeline import ingest, purge, query, reset, sources


def make_docs_dir(tmp_path, *filenames_and_contents):
    """Populate tmp_path with (filename, content) pairs and return its str path."""
    for filename, content in filenames_and_contents:
        (tmp_path / filename).write_text(content, encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def mock_chroma():
    """Patch chromadb so tests run without installing it or hitting disk."""
    collection = MagicMock()
    collection.count.return_value = 1000  # large enough for all query tests
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    client.get_collection.return_value = collection  # used by query()

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client), \
         patch("synapse_core.pipeline.embedding_functions.SentenceTransformerEmbeddingFunction"):
        yield collection


# --- ingest ---

def test_ingest_txt_file(mock_chroma, tmp_path):
    docs = make_docs_dir(tmp_path, ("hello.txt", "Hello world " * 10))
    ingest(source_dir=docs, db_path=str(tmp_path / "db"), verbose=False)
    assert mock_chroma.upsert.called


def test_ingest_multiple_files(mock_chroma, tmp_path):
    docs = make_docs_dir(
        tmp_path,
        ("a.txt", "Content of A " * 20),
        ("b.md", "Content of B " * 20),
    )
    ingest(source_dir=docs, db_path=str(tmp_path / "db"), verbose=False)
    assert mock_chroma.upsert.call_count == 2


def test_ingest_is_idempotent(mock_chroma, tmp_path):
    """Running ingest twice should call upsert both times (not insert)."""
    docs = make_docs_dir(tmp_path, ("doc.txt", "Some text " * 10))
    ingest(source_dir=docs, db_path=str(tmp_path / "db"), verbose=False)
    ingest(source_dir=docs, db_path=str(tmp_path / "db"), verbose=False)
    assert mock_chroma.upsert.call_count == 2


def test_ingest_skips_unsupported_files(mock_chroma, tmp_path):
    docs = make_docs_dir(tmp_path, ("image.png", "fake png data"))
    ingest(source_dir=docs, db_path=str(tmp_path / "db"), verbose=False)
    mock_chroma.upsert.assert_not_called()


def test_ingest_empty_file_is_skipped(mock_chroma, tmp_path):
    docs = make_docs_dir(tmp_path, ("empty.txt", ""))
    ingest(source_dir=docs, db_path=str(tmp_path / "db"), verbose=False)
    mock_chroma.upsert.assert_not_called()


def test_ingest_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest(source_dir=str(tmp_path / "nonexistent"), db_path=str(tmp_path / "db"))


def test_ingest_no_supported_files_verbose_false_no_output(mock_chroma, tmp_path, capsys):
    docs = make_docs_dir(tmp_path, ("image.png", "fake"))
    ingest(source_dir=docs, db_path=str(tmp_path / "db"), verbose=False)
    assert capsys.readouterr().out == ""


def test_ingest_skip_messages_suppressed_when_verbose_false(mock_chroma, tmp_path, capsys):
    """Empty file triggers [skip] path — must produce no output when verbose=False."""
    docs = make_docs_dir(tmp_path, ("empty.txt", ""))
    ingest(source_dir=docs, db_path=str(tmp_path / "db"), verbose=False)
    assert capsys.readouterr().out == ""


def test_upsert_payload_structure(mock_chroma, tmp_path):
    """Verify that ids, documents and metadatas are passed to upsert."""
    docs = make_docs_dir(tmp_path, ("test.txt", "word " * 100))
    ingest(source_dir=docs, db_path=str(tmp_path / "db"), verbose=False)

    call_kwargs = mock_chroma.upsert.call_args.kwargs
    assert "ids" in call_kwargs
    assert "documents" in call_kwargs
    assert "metadatas" in call_kwargs
    assert len(call_kwargs["ids"]) == len(call_kwargs["documents"])
    assert call_kwargs["metadatas"][0]["source_type"] == "file"


# --- purge ---

def test_purge_removes_stale_chunks(tmp_path):
    existing_file = tmp_path / "existing.txt"
    existing_file.write_text("hello", encoding="utf-8")

    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["id1", "id2"],
        "metadatas": [
            {"source": str(tmp_path / "deleted.txt"), "chunk": 0},  # stale
            {"source": str(existing_file), "chunk": 0},              # live
        ],
    }
    client = MagicMock()
    client.get_collection.return_value = collection

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client):
        deleted = purge(db_path=str(tmp_path / "db"), verbose=False)

    assert deleted == 1
    collection.delete.assert_called_once_with(ids=["id1"])


def test_purge_keeps_sqlite_chunks_when_db_exists(tmp_path):
    db_file = tmp_path / "data.db"
    db_file.write_text("", encoding="utf-8")  # just needs to exist on disk

    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["id1"],
        "metadatas": [{
            "source_type": "sqlite",
            "source": f"{db_file}::articles",
            "chunk": 0,
        }],
    }
    client = MagicMock()
    client.get_collection.return_value = collection

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client):
        deleted = purge(db_path=str(tmp_path / "db"), verbose=False)

    assert deleted == 0  # db file still exists — chunks must NOT be purged


def test_purge_removes_sqlite_chunks_when_db_missing(tmp_path):
    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["id1"],
        "metadatas": [{
            "source_type": "sqlite",
            "source": str(tmp_path / "gone.db::articles"),
            "chunk": 0,
        }],
    }
    client = MagicMock()
    client.get_collection.return_value = collection

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client):
        deleted = purge(db_path=str(tmp_path / "db"), verbose=False)

    assert deleted == 1  # db file is gone — chunks must be purged


def test_purge_nothing_when_all_exist(tmp_path):
    existing_file = tmp_path / "file.txt"
    existing_file.write_text("hello", encoding="utf-8")

    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["id1"],
        "metadatas": [{"source": str(existing_file), "chunk": 0}],
    }
    client = MagicMock()
    client.get_collection.return_value = collection

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client):
        deleted = purge(db_path=str(tmp_path / "db"), verbose=False)

    assert deleted == 0
    collection.delete.assert_not_called()


# --- reset ---

def test_reset_deletes_collection(tmp_path):
    client = MagicMock()
    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client):
        reset(db_path=str(tmp_path / "db"), verbose=False)
    client.delete_collection.assert_called_once_with(name="synapse")


# --- sources ---

def test_sources_returns_unique_sorted_files(tmp_path):
    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["id1", "id2", "id3"],
        "metadatas": [
            {"source": "/docs/b.txt", "chunk": 0},
            {"source": "/docs/a.txt", "chunk": 0},
            {"source": "/docs/b.txt", "chunk": 1},  # duplicate
        ],
    }
    client = MagicMock()
    client.get_collection.return_value = collection

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client):
        result = sources(db_path=str(tmp_path / "db"))

    assert result == ["/docs/a.txt", "/docs/b.txt"]


def test_sources_returns_empty_if_no_collection(tmp_path):
    client = MagicMock()
    client.get_collection.side_effect = ValueError("Collection not found")

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client):
        result = sources(db_path=str(tmp_path / "db"))

    assert result == []


# --- query ---

def test_query_returns_list_of_dicts(mock_chroma, tmp_path):
    mock_chroma.query.return_value = {
        "documents": [["chunk one", "chunk two"]],
        "metadatas": [[
            {"source": "/docs/a.txt", "source_type": "file", "chunk": 0},
            {"source": "/data.db::articles", "source_type": "sqlite", "chunk": 1},
        ]],
        "distances": [[0.1, 0.5]],
    }
    results = query(text="test query", db_path=str(tmp_path / "db"))
    assert len(results) == 2
    assert results[0]["text"] == "chunk one"
    assert results[0]["source"] == "/docs/a.txt"
    assert results[0]["source_type"] == "file"
    assert results[0]["chunk"] == 0
    assert results[0]["distance"] == 0.1
    assert results[0]["score"] == round(1 / 1.1, 4)
    assert results[1]["source_type"] == "sqlite"


def test_query_score_perfect_at_zero_distance(mock_chroma, tmp_path):
    mock_chroma.query.return_value = {
        "documents": [["exact match"]],
        "metadatas": [[{"source": "/a.txt", "chunk": 0}]],
        "distances": [[0.0]],
    }
    results = query(text="test", db_path=str(tmp_path / "db"))
    assert results[0]["score"] == 1.0


def test_query_empty_collection_returns_empty(mock_chroma, tmp_path):
    mock_chroma.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    results = query(text="test", db_path=str(tmp_path / "db"))
    assert results == []


def test_query_raises_if_collection_not_found(tmp_path):
    client = MagicMock()
    client.get_collection.side_effect = ValueError("Collection not found")

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client), \
         patch("synapse_core.pipeline.embedding_functions.SentenceTransformerEmbeddingFunction"):
        with pytest.raises(ValueError, match="ingest()"):
            query(text="test", db_path=str(tmp_path / "db"))


# --- incremental ingestion ---

def test_incremental_skips_unchanged_file(tmp_path):
    """File with same hash as stored → upsert must NOT be called."""
    content = "Hello world " * 20
    docs = make_docs_dir(tmp_path, ("doc.txt", content))
    file_path = tmp_path / "doc.txt"

    import hashlib
    current_hash = hashlib.sha256(file_path.read_bytes(), usedforsecurity=False).hexdigest()

    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["id0"],
        "metadatas": [{"source_type": "file", "source": str(file_path), "chunk": 0, "file_hash": current_hash}],
    }
    client = MagicMock()
    client.get_or_create_collection.return_value = collection

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client), \
         patch("synapse_core.pipeline.embedding_functions.SentenceTransformerEmbeddingFunction"):
        ingest(source_dir=docs, db_path=str(tmp_path / "db"), incremental=True, verbose=False)

    collection.upsert.assert_not_called()


def test_incremental_reingests_changed_file(tmp_path):
    """File with different stored hash → delete old chunks + upsert new ones."""
    content = "Hello world " * 20
    docs = make_docs_dir(tmp_path, ("doc.txt", content))
    file_path = tmp_path / "doc.txt"

    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["old_id0"],
        "metadatas": [{"source_type": "file", "source": str(file_path), "chunk": 0, "file_hash": "stale_hash"}],
    }
    client = MagicMock()
    client.get_or_create_collection.return_value = collection

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client), \
         patch("synapse_core.pipeline.embedding_functions.SentenceTransformerEmbeddingFunction"):
        ingest(source_dir=docs, db_path=str(tmp_path / "db"), incremental=True, verbose=False)

    collection.delete.assert_called_once_with(ids=["old_id0"])
    assert collection.upsert.called


def test_incremental_ingests_new_file(tmp_path):
    """File with no existing chunks → ingest as normal."""
    docs = make_docs_dir(tmp_path, ("new.txt", "Brand new content " * 10))

    collection = MagicMock()
    collection.get.return_value = {"ids": [], "metadatas": []}
    client = MagicMock()
    client.get_or_create_collection.return_value = collection

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client), \
         patch("synapse_core.pipeline.embedding_functions.SentenceTransformerEmbeddingFunction"):
        ingest(source_dir=docs, db_path=str(tmp_path / "db"), incremental=True, verbose=False)

    collection.delete.assert_not_called()
    assert collection.upsert.called


def test_incremental_stores_hash_in_metadata(tmp_path):
    """Metadata must contain file_hash when incremental=True."""
    docs = make_docs_dir(tmp_path, ("doc.txt", "word " * 50))

    collection = MagicMock()
    collection.get.return_value = {"ids": [], "metadatas": []}
    client = MagicMock()
    client.get_or_create_collection.return_value = collection

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client), \
         patch("synapse_core.pipeline.embedding_functions.SentenceTransformerEmbeddingFunction"):
        ingest(source_dir=docs, db_path=str(tmp_path / "db"), incremental=True, verbose=False)

    metadatas = collection.upsert.call_args.kwargs["metadatas"]
    assert all("file_hash" in m for m in metadatas)


def test_non_incremental_does_not_store_hash(mock_chroma, tmp_path):
    """Default (incremental=False) must NOT add file_hash to metadata."""
    docs = make_docs_dir(tmp_path, ("doc.txt", "word " * 50))
    ingest(source_dir=docs, db_path=str(tmp_path / "db"), verbose=False)

    metadatas = mock_chroma.upsert.call_args.kwargs["metadatas"]
    assert all("file_hash" not in m for m in metadatas)


# --- sentence chunking ---

def test_ingest_sentence_chunking(mock_chroma, tmp_path):
    """chunking='sentence' must still produce chunks and call upsert."""
    pytest.importorskip("nltk")
    text = "The sky is blue. The grass is green. " * 30
    docs = make_docs_dir(tmp_path, ("doc.txt", text))
    ingest(source_dir=docs, db_path=str(tmp_path / "db"), chunking="sentence", verbose=False)
    assert mock_chroma.upsert.called


# --- query: n_results clamping ---

def test_query_clamps_n_results_to_collection_size(tmp_path):
    """query() must not crash when n_results > number of documents in collection."""
    collection = MagicMock()
    collection.count.return_value = 2
    collection.query.return_value = {
        "documents": [["chunk one", "chunk two"]],
        "metadatas": [[
            {"source": "/a.txt", "source_type": "file", "chunk": 0},
            {"source": "/b.txt", "source_type": "file", "chunk": 0},
        ]],
        "distances": [[0.1, 0.2]],
    }
    client = MagicMock()
    client.get_collection.return_value = collection

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client), \
         patch("synapse_core.pipeline.embedding_functions.SentenceTransformerEmbeddingFunction"):
        results = query(text="test", db_path=str(tmp_path / "db"), n_results=10)

    called_n = collection.query.call_args.kwargs["n_results"]
    assert called_n == 2
    assert len(results) == 2


def test_query_returns_empty_on_empty_collection(tmp_path):
    """query() must return [] immediately when the collection is empty."""
    collection = MagicMock()
    collection.count.return_value = 0
    client = MagicMock()
    client.get_collection.return_value = collection

    with patch("synapse_core.pipeline.chromadb.PersistentClient", return_value=client), \
         patch("synapse_core.pipeline.embedding_functions.SentenceTransformerEmbeddingFunction"):
        results = query(text="test", db_path=str(tmp_path / "db"))

    collection.query.assert_not_called()
    assert results == []

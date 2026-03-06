import hashlib
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from .chunker import chunk_text
from .extractors import extract, is_supported


def _make_id(file_path: Path, source_dir: Path, chunk_index: int) -> str:
    """Stable unique ID based on relative path — portable across machine moves."""
    try:
        rel = file_path.relative_to(source_dir)
    except ValueError:
        rel = file_path.resolve()
    key = f"{rel}::{chunk_index}"
    return hashlib.md5(key.encode()).hexdigest()


def _get_collection(db_path: str, collection_name: str, embedding_model: str):
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    return client.get_or_create_collection(
        name=collection_name, embedding_function=ef  # type: ignore[arg-type]
    )


def ingest(
    source_dir: str = "./docs",
    db_path: str = "./synapse_db",
    collection_name: str = "synapse",
    chunk_size: int = 1000,
    overlap: int = 200,
    min_chunk_size: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
    verbose: bool = True,
) -> None:
    """
    Scan source_dir for supported files, extract text, chunk it,
    embed it and store everything in a local ChromaDB collection.

    Args:
        source_dir:       Directory containing files to ingest.
        db_path:          Path where ChromaDB persists data.
        collection_name:  Name of the ChromaDB collection.
        chunk_size:       Target character count per chunk.
        overlap:          Character overlap between consecutive chunks.
        min_chunk_size:   Discard chunks shorter than this (chars).
        embedding_model:  SentenceTransformer model name.
        verbose:          Print progress to stdout.
    """
    source = Path(source_dir)
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    collection = _get_collection(db_path, collection_name, embedding_model)

    files = [f for f in source.rglob("*") if f.is_file() and is_supported(f)]
    if not files:
        if verbose:
            print(f"No supported files found in {source}")
        return

    for file_path in files:
        if verbose:
            print(f"Ingesting: {file_path.name}")
        try:
            text = extract(file_path)
        except Exception as e:
            print(f"  [skip] {file_path.name}: {e}")
            continue

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )
        if not chunks:
            print(f"  [skip] {file_path.name}: no text extracted")
            continue

        ids = [_make_id(file_path, source, i) for i in range(len(chunks))]
        metadatas = [
            {"source_type": "file", "source": str(file_path.resolve()), "chunk": i}
            for i in range(len(chunks))
        ]

        # Upsert so re-running is idempotent
        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)  # type: ignore[arg-type]

        if verbose:
            print(f"  -> {len(chunks)} chunks stored")

    if verbose:
        print(f"\nDone. Collection '{collection_name}' in '{db_path}'")


def _source_exists(meta: dict) -> bool:
    """Return True if the chunk's source still exists on disk.

    File sources: absolute path — check directly.
    SQLite sources: stored as "/abs/path/to/db::table" — check only the db file part.
    """
    source = meta.get("source", "")
    if meta.get("source_type") == "sqlite":
        db_file = source.split("::")[0]
        return Path(db_file).exists()
    return Path(source).exists()


def purge(
    db_path: str = "./synapse_db",
    collection_name: str = "synapse",
    verbose: bool = True,
) -> int:
    """
    Remove chunks from ChromaDB whose source file no longer exists on disk.

    Returns the number of chunks deleted.
    """
    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        if verbose:
            print(f"Collection '{collection_name}' not found.")
        return 0

    results = collection.get(include=["metadatas"])
    stale_ids = [
        id_
        for id_, meta in zip(results["ids"], results["metadatas"])  # type: ignore[arg-type]
        if not _source_exists(meta)
    ]

    if stale_ids:
        collection.delete(ids=stale_ids)
        if verbose:
            print(f"Purged {len(stale_ids)} stale chunk(s).")
    elif verbose:
        print("Nothing to purge — all sources still exist.")

    return len(stale_ids)


def reset(
    db_path: str = "./synapse_db",
    collection_name: str = "synapse",
    verbose: bool = True,
) -> None:
    """Delete the entire ChromaDB collection."""
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(name=collection_name)
        if verbose:
            print(f"Collection '{collection_name}' deleted.")
    except ValueError:
        if verbose:
            print(f"Collection '{collection_name}' not found.")


def sources(
    db_path: str = "./synapse_db",
    collection_name: str = "synapse",
) -> List[str]:
    """Return a sorted list of unique source file paths stored in the collection."""
    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        return []

    results = collection.get(include=["metadatas"])
    seen = set()
    unique = []
    for meta in results["metadatas"]:  # type: ignore[union-attr]
        src = meta.get("source", "")
        if src and src not in seen:
            seen.add(src)
            unique.append(src)
    return sorted(unique)

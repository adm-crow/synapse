import hashlib
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from .chunker import chunk_text
from .extractors import extract, is_supported
from .logger import logger


def _make_id(file_path: Path, source_dir: Path, chunk_index: int) -> str:
    """Stable unique ID based on relative path — portable across machine moves."""
    try:
        rel = file_path.relative_to(source_dir)
    except ValueError:
        rel = file_path.resolve()
    key = f"{rel}::{chunk_index}"
    return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()


def _get_collection(db_path: str, collection_name: str, embedding_model: str, create: bool = True):
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    if create:
        return client.get_or_create_collection(
            name=collection_name, embedding_function=ef  # type: ignore[arg-type]
        )
    return client.get_collection(
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

    files = [f for f in source.rglob("*") if f.is_file() and is_supported(f)]
    if not files:
        if verbose:
            logger.info("No supported files found in %s", source)
        return

    collection = _get_collection(db_path, collection_name, embedding_model)

    for file_path in files:
        if verbose:
            logger.info("Ingesting: %s", file_path.name)
        try:
            text = extract(file_path)
        except Exception as e:
            if verbose:
                logger.warning("[skip] %s: %s", file_path.name, e)
            continue

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )
        if not chunks:
            if verbose:
                logger.warning("[skip] %s: no text extracted", file_path.name)
            continue

        ids = [_make_id(file_path, source, i) for i in range(len(chunks))]
        metadatas = [
            {"source_type": "file", "source": str(file_path.resolve()), "chunk": i}
            for i in range(len(chunks))
        ]

        # Upsert so re-running is idempotent
        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)  # type: ignore[arg-type]

        if verbose:
            logger.info("  -> %d chunks stored", len(chunks))

    if verbose:
        logger.info("Done. Collection '%s' in '%s'", collection_name, db_path)


def query(
    text: str,
    db_path: str = "./synapse_db",
    collection_name: str = "synapse",
    n_results: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> List[dict]:
    """
    Semantic search over the ChromaDB collection.

    Args:
        text:             Query string.
        db_path:          Path to the ChromaDB directory.
        collection_name:  Name of the ChromaDB collection.
        n_results:        Maximum number of results to return.
        embedding_model:  SentenceTransformer model name (must match ingest).

    Returns:
        List of dicts sorted by relevance (highest score first), each with:
        - text:        chunk content
        - source:      origin file path
        - source_type: "file" or "sqlite"
        - score:       relevance score 0–1 (1 = perfect match)
        - distance:    raw ChromaDB L2 distance (lower = closer)
        - chunk:       chunk index within the source document
    """
    try:
        collection = _get_collection(db_path, collection_name, embedding_model, create=False)
    except ValueError:
        raise ValueError(
            f"Collection '{collection_name}' not found in '{db_path}' — run ingest() first."
        )
    results = collection.query(
        query_texts=[text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    documents = results["documents"][0]  # type: ignore[index]
    metadatas = results["metadatas"][0]  # type: ignore[index]
    distances = results["distances"][0]  # type: ignore[index]
    return [
        {
            "text": doc,
            "source": meta.get("source", ""),
            "source_type": meta.get("source_type", "file"),
            "score": round(1 / (1 + dist), 4),
            "distance": round(dist, 4),
            "chunk": meta.get("chunk", 0),
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


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
            logger.warning("Collection '%s' not found.", collection_name)
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
            logger.info("Purged %d stale chunk(s).", len(stale_ids))
    elif verbose:
        logger.info("Nothing to purge — all sources still exist.")

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
            logger.info("Collection '%s' deleted.", collection_name)
    except ValueError:
        if verbose:
            logger.warning("Collection '%s' not found.", collection_name)


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

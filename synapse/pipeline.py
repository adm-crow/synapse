import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from .chunker import chunk_text
from .extractors import extract, is_supported


def _make_id(file_path: Path, chunk_index: int) -> str:
    """Stable unique ID for a chunk: hash of filepath + chunk index."""
    key = f"{file_path.resolve()}::{chunk_index}"
    return hashlib.md5(key.encode()).hexdigest()


def ingest(
    source_dir: str = "./docs",
    db_path: str = "./synapse_db",
    collection_name: str = "synapse",
    chunk_size: int = 500,
    overlap: int = 50,
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
        chunk_size:       Approximate character count per chunk.
        overlap:          Character overlap between consecutive chunks.
        embedding_model:  SentenceTransformer model name.
        verbose:          Print progress to stdout.
    """
    source = Path(source_dir)
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    # ChromaDB client + sentence-transformer embedding function
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=ef
    )

    files = [f for f in source.rglob("*") if f.is_file() and is_supported(f)]
    if not files:
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

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            print(f"  [skip] {file_path.name}: no text extracted")
            continue

        ids = [_make_id(file_path, i) for i in range(len(chunks))]
        metadatas = [
            {"source": str(file_path.resolve()), "chunk": i}
            for i in range(len(chunks))
        ]

        # Upsert so re-running is idempotent
        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

        if verbose:
            print(f"  -> {len(chunks)} chunks stored")

    if verbose:
        print(f"\nDone. Collection '{collection_name}' in '{db_path}'")

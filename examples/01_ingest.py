"""
Example 1 — Ingestion
=====================

Populate a ChromaDB collection from two sources:
  - a local folder of files
  - a SQLite database table

Run this once. Examples 02, 03 and 04 all read from the same collection.
"""

from synapse_core import ingest, ingest_sqlite

if __name__ == "__main__":
    # --- files ---------------------------------------------------------------
    # Recursively scans ./docs for .txt .md .pdf .docx .csv .json .jsonl files,
    # extracts text, chunks it, embeds it and upserts into ChromaDB.

    ingest(
        source_dir="./docs",       # folder to scan (any path works)
        db_path="./synapse_db",    # ChromaDB will be created here
        collection_name="synapse",
        chunk_size=1000,
        overlap=200,
        # incremental=True,        # skip files whose content hasn't changed (SHA-256)
        # chunking="sentence",     # sentence-aware splitting (pip install synapse-core[sentence])
        verbose=True,
    )

    # --- SQLite table ---------------------------------------------------------
    # Rows are serialised to "col: value | col: value | ..." then go through
    # the same chunker and embedder. Both sources end up in the same collection.

    ingest_sqlite(
        db_path="./data.db",
        table="articles",
        # columns=["title", "body"],        # optional: restrict which columns to embed
        # row_template="{title}: {body}",   # optional: custom row format
        # chunking="sentence",              # sentence-aware splitting
        chroma_path="./synapse_db",
        collection_name="synapse",
        verbose=True,
    )

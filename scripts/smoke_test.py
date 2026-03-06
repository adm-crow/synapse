"""
End-to-end smoke test — runs the real pipeline (no mocks).

Usage:
    uv run python scripts/smoke_test.py

What it validates:
  - ingest() scans a real folder, extracts text, embeds and stores in ChromaDB
  - ingest_sqlite() ingests real rows from a real SQLite table
  - ChromaDB query returns results for both sources
  - sources() lists both sources
  - purge() correctly keeps live chunks and removes stale ones
  - reset() wipes the collection

This downloads the embedding model (~90 MB) on first run.
Everything is written to a temp directory and cleaned up afterwards.
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: allow running from project root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.utils import embedding_functions

from synapse_core import ingest, ingest_sqlite, purge, reset, sources

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {label}")
    else:
        msg = f"  {FAIL}  {label}" + (f" — {detail}" if detail else "")
        print(msg)
        errors.append(label)


# ---------------------------------------------------------------------------
# Setup: temp workspace
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    tmp = Path(tmp)
    docs_dir = tmp / "docs"
    docs_dir.mkdir()
    db_path = str(tmp / "synapse_db")
    sqlite_path = str(tmp / "data.db")

    print("\n=== 1. Preparing test fixtures ===")

    # Write sample files
    (docs_dir / "policy.txt").write_text(
        "Refund policy: customers may return items within 30 days for a full refund. "
        "No questions asked. Contact support@example.com to initiate. " * 10,
        encoding="utf-8",
    )
    (docs_dir / "faq.md").write_text(
        "# FAQ\n\n"
        "## Shipping\nWe ship worldwide. Standard delivery takes 5-7 business days. " * 8,
        encoding="utf-8",
    )
    (docs_dir / "ignore.png").write_bytes(b"\x89PNG")  # unsupported — must be skipped

    # Create SQLite DB with sample data
    conn = sqlite3.connect(sqlite_path)
    conn.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, body TEXT)")
    conn.executemany(
        "INSERT INTO articles VALUES (?, ?, ?)",
        [
            (1, "Return policy", "Items can be returned within 30 days of purchase. " * 8),
            (2, "Shipping info", "Express delivery is available for an extra fee. " * 8),
        ],
    )
    conn.commit()
    conn.close()
    print("  Created 2 files + 1 SQLite table (2 rows)")

    # ---------------------------------------------------------------------------
    print("\n=== 2. ingest() — files ===")
    ingest(source_dir=str(docs_dir), db_path=db_path, verbose=True)

    # ---------------------------------------------------------------------------
    print("\n=== 3. ingest_sqlite() ===")
    ingest_sqlite(
        db_path=sqlite_path, table="articles",
        chroma_path=db_path, verbose=True,
    )

    # ---------------------------------------------------------------------------
    print("\n=== 4. Verifying ChromaDB contents ===")
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(  # type: ignore[attr-defined]
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection("synapse", embedding_function=ef)  # type: ignore[arg-type]
    total = collection.count()
    check("Collection has chunks", total > 0, f"count={total}")

    results = collection.query(query_texts=["refund policy"], n_results=3)
    docs_found = results["documents"][0]  # type: ignore[index]
    metas_found = results["metadatas"][0]  # type: ignore[index]
    check("Query returns results", len(docs_found) > 0)

    source_types = {m["source_type"] for m in metas_found}
    check(
        "Results include both file and sqlite sources",
        source_types == {"file", "sqlite"},
        f"found source_types={source_types}",
    )

    # ---------------------------------------------------------------------------
    print("\n=== 5. sources() ===")
    all_sources = sources(db_path=db_path)
    check("sources() returns entries", len(all_sources) > 0)
    file_sources = [s for s in all_sources if "::" not in s]
    sqlite_sources = [s for s in all_sources if "::" in s]
    check("sources() includes file paths", len(file_sources) > 0, str(file_sources))
    check("sources() includes sqlite source", len(sqlite_sources) > 0, str(sqlite_sources))
    print(f"  All sources: {all_sources}")

    # ---------------------------------------------------------------------------
    print("\n=== 6. purge() ===")
    # Delete one file from disk — its chunks should be purged
    (docs_dir / "faq.md").unlink()
    deleted = purge(db_path=db_path, verbose=True)
    check("purge() removed stale file chunks", deleted > 0, f"deleted={deleted}")

    remaining = sources(db_path=db_path)
    faq_still_present = any("faq.md" in s for s in remaining)
    check("purge() did not remove live sources", not faq_still_present)

    sqlite_still_present = any("::" in s for s in remaining)
    check("purge() kept sqlite chunks (db file still exists)", sqlite_still_present)

    # ---------------------------------------------------------------------------
    print("\n=== 7. reset() ===")
    reset(db_path=db_path, verbose=True)
    try:
        client2 = chromadb.PersistentClient(path=db_path)
        client2.get_collection("synapse")
        check("reset() wiped the collection", False, "collection still exists")
    except Exception:
        check("reset() wiped the collection", True)

    # ---------------------------------------------------------------------------
    print("\n=== 8. Idempotency — re-ingest produces same chunk count ===")
    # Re-create faq.md (was deleted in step 6) and start from a clean db
    (docs_dir / "faq.md").write_text(
        "# FAQ\n\nWe ship worldwide. Standard delivery takes 5-7 business days. " * 8,
        encoding="utf-8",
    )
    ingest(source_dir=str(docs_dir), db_path=db_path, verbose=False)
    # Need a fresh client reference — the old one pointed at the deleted collection
    fresh_client = chromadb.PersistentClient(path=db_path)
    fresh_col = fresh_client.get_collection("synapse", embedding_function=ef)  # type: ignore[arg-type]
    count_first = fresh_col.count()
    ingest(source_dir=str(docs_dir), db_path=db_path, verbose=False)
    count_second = fresh_col.count()
    check(
        "Re-ingesting same files produces same chunk count",
        count_first == count_second,
        f"first={count_first} second={count_second}",
    )

# ---------------------------------------------------------------------------
print()
if errors:
    print(f"FAILED — {len(errors)} check(s) did not pass:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"All checks passed.")

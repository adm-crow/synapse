"""
Example 4 — Collection management
====================================

Demonstrates the three collection utilities:

  sources() — list every ingested source (files and SQLite tables)
  purge()   — remove chunks whose source file no longer exists on disk
  reset()   — wipe the entire collection (destructive, use with care)
"""

from synapse import purge, reset, sources

DB_PATH = "./synapse_db"
COLLECTION = "synapse"

# --- sources() -----------------------------------------------------------
# Returns a sorted list of unique source paths currently in the collection.

print("=== Ingested sources ===")
for path in sources(db_path=DB_PATH, collection_name=COLLECTION):
    print(f"  {path}")

# Example output:
#   /home/user/docs/company_policy.pdf
#   /home/user/docs/product_faq.txt
#   /home/user/data.db::articles


# --- purge() -------------------------------------------------------------
# When source files are deleted from disk, their chunks remain in ChromaDB
# until you call purge(). Safe to call repeatedly — it only removes chunks
# whose source path no longer exists.

print("\n=== Purging stale chunks ===")
deleted = purge(db_path=DB_PATH, collection_name=COLLECTION, verbose=True)
print(f"Removed {deleted} stale chunk(s).")


# --- reset() -------------------------------------------------------------
# Deletes the entire collection. All vectors and metadata are wiped.
# Re-run 01_ingest.py afterwards to start fresh.

# Uncomment to use:
# print("\n=== Resetting collection ===")
# reset(db_path=DB_PATH, collection_name=COLLECTION, verbose=True)

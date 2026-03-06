"""
Example 2 — Query the collection
==================================

Runs a semantic search against the collection populated by 01_ingest.py.
Results are sorted by relevance score (1 = perfect match, 0 = unrelated).
"""

from synapse_core import query

if __name__ == "__main__":
    QUESTION = "What is the refund policy?"

    results = query(
        text=QUESTION,
        db_path="./synapse_db",
        collection_name="synapse",
        n_results=4,
    )

    print(f"Query: {QUESTION!r}
")
    for i, r in enumerate(results):
        print(f"[{i + 1}] score={r['score']:.3f}  source={r['source']}  chunk={r['chunk']}")
        print(f"    {r['text'][:300]}")
        print()

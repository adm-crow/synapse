"""
Example 2 — Query ChromaDB directly
=====================================

Connects to the collection populated by 01_ingest.py and runs a semantic
search query. Returns the top N chunks with their relevance distances.

ChromaDB returns L2 distances (lower = more similar). To get a rough
similarity score, use: score = 1 / (1 + distance)
"""

import chromadb
from chromadb.utils import embedding_functions

QUESTION = "What is the refund policy?"
N_RESULTS = 4

client = chromadb.PersistentClient(path="./synapse_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(  # type: ignore[attr-defined]
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection("synapse", embedding_function=ef)  # type: ignore[arg-type]

results = collection.query(
    query_texts=[QUESTION],
    n_results=N_RESULTS,
    include=["documents", "metadatas", "distances"],
)

documents = results["documents"][0]  # type: ignore[index]
metadatas = results["metadatas"][0]  # type: ignore[index]
distances = results["distances"][0]  # type: ignore[index]

print(f"Query: {QUESTION!r}\n")
for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
    score = 1 / (1 + dist)
    print(f"[{i + 1}] score={score:.3f}  source={meta['source']}  chunk={meta['chunk']}")
    print(f"    {doc[:300]}")
    print()

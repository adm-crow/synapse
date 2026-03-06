"""
Example 3 — Connecting a RAG agent
====================================

Shows the full RAG pattern:
  1. Retrieve relevant chunks from ChromaDB (synapse handles this)
  2. Build a prompt with the retrieved context
  3. Call your LLM of choice

The `call_llm` function below is a stub — replace it with any real LLM:
  - Anthropic:  anthropic.Anthropic().messages.create(...)
  - OpenAI:     openai.OpenAI().chat.completions.create(...)
  - Ollama:     requests.post("http://localhost:11434/api/generate", ...)
"""

import chromadb
from chromadb.utils import embedding_functions

# --- ChromaDB setup -------------------------------------------------------

client = chromadb.PersistentClient(path="./synapse_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(  # type: ignore[attr-defined]
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection("synapse", embedding_function=ef)  # type: ignore[arg-type]


# --- LLM stub (replace with your real provider) --------------------------

def call_llm(prompt: str) -> str:
    """Stub — replace with your actual LLM call."""
    return f"[LLM stub] received {len(prompt)} chars of context + question."


# --- RAG agent ------------------------------------------------------------

def ask(question: str, n_results: int = 4) -> str:
    # 1. Retrieve the most relevant chunks
    results = collection.query(query_texts=[question], n_results=n_results)
    chunks = results["documents"][0]  # type: ignore[index]
    metas = results["metadatas"][0]   # type: ignore[index]

    # 2. Build context block with source attribution
    context_parts = []
    for chunk, meta in zip(chunks, metas):
        source = meta.get("source", "unknown")
        context_parts.append(f"[source: {source}]\n{chunk}")
    context = "\n\n---\n\n".join(context_parts)

    # 3. Build prompt and call the LLM
    prompt = (
        f"Use the following context to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {question}"
    )
    return call_llm(prompt)


# --- Run ------------------------------------------------------------------

if __name__ == "__main__":
    answer = ask("What is the refund policy?")
    print(answer)

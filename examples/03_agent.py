"""
Example 3 — Connecting a RAG agent
====================================

Shows the full RAG pattern:
  1. Retrieve relevant chunks with synapse_core.query()
  2. Build a prompt with the retrieved context
  3. Call your LLM of choice

The `call_llm` function below is a stub — replace it with any real LLM:
  - Anthropic:  anthropic.Anthropic().messages.create(...)
  - OpenAI:     openai.OpenAI().chat.completions.create(...)
  - Ollama:     requests.post("http://localhost:11434/api/generate", ...)
"""

from synapse_core import query


# --- LLM stub (replace with your real provider) --------------------------

def call_llm(prompt: str) -> str:
    """Stub — replace with your actual LLM call."""
    return f"[LLM stub] received {len(prompt)} chars of context + question."


# --- RAG agent ------------------------------------------------------------

def ask(question: str, n_results: int = 4) -> str:
    # 1. Retrieve the most relevant chunks
    results = query(text=question, n_results=n_results)

    # 2. Build context block with source attribution
    context_parts = [
        f"[source: {r['source']}]\n{r['text']}"
        for r in results
    ]
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

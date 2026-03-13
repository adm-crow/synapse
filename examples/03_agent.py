"""
Example 3 — Connecting a RAG agent (Anthropic SDK)
====================================================

Full RAG pattern using the Anthropic SDK:
  1. Ingest your docs once with synapse
  2. On every question, retrieve the most relevant chunks
  3. Pass the chunks as context to Claude

Setup:
  pip install synapse-core anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."

For other LLM providers, replace the `ask()` body:
  - OpenAI:  openai.OpenAI().chat.completions.create(...)
  - Ollama:  requests.post("http://localhost:11434/api/generate", ...)
"""

import anthropic

from synapse_core import ingest, query

# --- Step 1: ingest your docs once -----------------------------------------
# Comment this out after the first run — the collection persists on disk.
ingest("./docs")

# --- Step 2: set up the Anthropic client ------------------------------------
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment


# --- Step 3: RAG-powered ask function ---------------------------------------

def ask(question: str, n_results: int = 4) -> str:
    # Retrieve the most relevant chunks from your indexed docs
    chunks = query(question, n_results=n_results)

    # Build context block with source attribution
    context = "\n\n".join(
        f"[source: {r['source']}]\n{r['text']}"
        for r in chunks
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=(
            "You are a helpful assistant. "
            "Answer the user's question using ONLY the context below. "
            "If the answer is not in the context, say so.\n\n"
            f"CONTEXT:\n{context}"
        ),
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


# --- Run --------------------------------------------------------------------

if __name__ == "__main__":
    answer = ask("What is the refund policy?")
    print(answer)

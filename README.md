<div align="center">
  <img src="logo.svg" alt="Synapse" width="120" /><br/><br/>

[![CI](https://github.com/adm-crow/synapse/actions/workflows/ci.yml/badge.svg)](https://github.com/adm-crow/synapse/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-21%20passing-brightgreen?style=flat-square)](tests/)
[![build](https://img.shields.io/github/actions/workflow/status/adm-crow/synapse/ci.yml?branch=main&style=flat-square&label=build)](https://github.com/adm-crow/synapse/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-brightgreen?style=flat-square)](LICENSE)
[![pypi](https://img.shields.io/pypi/v/synapse-rag?style=flat-square&label=pypi)](https://pypi.org/project/synapse-rag/)

</div>

**synapse** is a local-first Python library for building file-based RAG pipelines — drop files into a folder, run the pipeline, and let any AI agent search your knowledge base with vector queries.

```
Your files  ──►  Extractor  ──►  Chunker  ──►  Embedder  ──►  ChromaDB
```

| | Feature | Details |
|---|---|---|
| 📄 | **5 formats** | `txt`, `md`, `csv`, `pdf`, `docx` — auto-detected by extension |
| ✂️ | **Smart chunking** | Sliding window with configurable size and overlap |
| 🧠 | **Local embeddings** | `sentence-transformers` — no API key, runs fully offline |
| 💾 | **ChromaDB** | Persistent vector store, zero config |
| 🔁 | **Idempotent** | Re-run safely — chunks are upserted, never duplicated |
| 🤖 | **Agent agnostic** | Works with LangChain, LlamaIndex, or any custom agent |

---

## 📦 Installation

```bash
pip install -e .
# or
uv add synapse
```

Includes everything out of the box: `txt`, `md`, `csv`, `pdf`, `docx`, embeddings, ChromaDB.

---

## 🚀 Quick start

```python
from synapse import ingest

ingest("./docs")
```

```
Ingesting: company_policy.pdf   ->  12 chunks stored
Ingesting: product_faq.txt      ->   8 chunks stored
Ingesting: meeting_notes.docx   ->   5 chunks stored

Done. Collection 'synapse' in './synapse_db'
```

> [!TIP]
> All parameters have sensible defaults. `ingest()` with no arguments scans `./docs` and stores everything in `./synapse_db`.

<details>
<summary>► See all <code>ingest()</code> options</summary>

```python
ingest(
    source_dir      = "./docs",             # folder to scan (recursive)
    db_path         = "./synapse_db",       # ChromaDB persistence path
    collection_name = "synapse",            # collection name
    chunk_size      = 1000,                 # characters per chunk
    overlap         = 200,                  # overlap between consecutive chunks
    embedding_model = "all-MiniLM-L6-v2",  # any SentenceTransformer model name
    verbose         = True,                 # print progress to stdout
)
```

</details>

---

## 🔌 Connecting to an AI agent

synapse handles the **ingestion** half of RAG. Wire the ChromaDB collection to any LLM to build a complete agent:

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./synapse_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
collection = client.get_collection("synapse", embedding_function=ef)

def ask(question: str) -> str:
    results = collection.query(query_texts=[question], n_results=4)
    context = "\n\n".join(results["documents"][0])
    # pass context to your LLM of choice
    return context

print(ask("What is the refund policy?"))
```

> [!IMPORTANT]
> synapse is model-agnostic — it only provides retrieved chunks as context. The same pattern works with Anthropic, OpenAI, Ollama, Mistral, or any other LLM.

> [!NOTE]
> Each chunk includes `source` (absolute file path) and `chunk` (index) in its metadata, so your agent always knows where an answer came from.

<details>
<summary>► Need to run ingest from an async context?</summary>

```python
import asyncio
from synapse import ingest

async def main():
    await asyncio.to_thread(ingest, "./docs")

asyncio.run(main())
```

`ingest()` is synchronous by design — the bottleneck is CPU-bound embedding, not I/O. Use `asyncio.to_thread` to avoid blocking an event loop.

</details>

---

## 📄 Supported formats

| Format | Extensions |
|---|---|
| Plain text | `.txt` `.md` |
| PDF | `.pdf` |
| Word document | `.docx` `.doc` |
| Spreadsheet | `.csv` |

---

## 🏗️ Architecture

```
synapse/
├── pipeline.py         ← ingest() orchestrator
├── extractors.py       ← file extension → raw text
└── chunker.py          ← raw text → overlapping chunks
```

The central flow through every stage:

```
File on disk
    │  extractors.py  →  raw text string
    │  chunker.py     →  list of chunk strings
    │  ChromaDB ef    →  vectors (sentence-transformers)
    ▼
chromadb.PersistentClient  →  upsert(documents, ids, metadatas)
```

---

## 🧪 Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

21 tests — chunker, extractors, and pipeline (ChromaDB mocked, fast).

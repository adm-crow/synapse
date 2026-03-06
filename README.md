<div align="center">
  <img src="logo.svg" alt="Synapse" width="120" /><br/><br/>

# ⚡Synapse

[![CI](https://github.com/adm-crow/synapse/actions/workflows/ci.yml/badge.svg)](https://github.com/adm-crow/synapse/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-41%20passing-brightgreen?style=flat-square)](tests/)
[![build](https://img.shields.io/github/actions/workflow/status/adm-crow/synapse/ci.yml?branch=main&style=flat-square&label=build)](https://github.com/adm-crow/synapse/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-brightgreen?style=flat-square)](LICENSE)
[![pypi](https://img.shields.io/pypi/v/synapse-rag?style=flat-square&label=pypi)](https://pypi.org/project/synapse-rag/)

</div>

**synapse** is a local-first Python library for building multi-source RAG pipelines — ingest files and databases into a single vector store and let any AI agent search across all your knowledge with one query.

```
Files      ──►  Extractor       ──►  Chunker  ──►  Embedder  ──►  ChromaDB
SQLite DB  ──►  Row serializer  ──►  Chunker  ──►  Embedder  ──►  (same collection)
```

| | Feature | Details |
|---|---|---|
| 📄 | **7 file formats** | `txt`, `md`, `csv`, `pdf`, `docx`, `json`, `jsonl` — auto-detected by extension |
| 🗄️ | **SQLite ingestion** | Ingest table records alongside files into the same collection |
| ✂️ | **Smart chunking** | Word-boundary aware, configurable size, overlap, and minimum chunk size |
| 🧠 | **Local embeddings** | `sentence-transformers` — no API key, runs fully offline |
| 💾 | **ChromaDB** | Persistent vector store, zero config |
| 🔁 | **Idempotent** | Re-run safely — chunks are upserted, never duplicated |
| 🧹 | **Maintainable** | `purge()`, `reset()`, `sources()` to manage your collection over time |
| 🤖 | **Agent agnostic** | Works with LangChain, LlamaIndex, or any custom agent |

---

## 📦 Installation

```bash
pip install synapse
```
or
```bash
uv add synapse
```

Includes everything out of the box: `txt`, `md`, `csv`, `pdf`, `docx`, `json`, `jsonl`, embeddings, ChromaDB.

---

## 🚀 Quick start

**From files:**

```python
from synapse import ingest

ingest("./my_documents")
```

```
Ingesting: company_policy.pdf   ->  12 chunks stored
Ingesting: product_faq.txt      ->   8 chunks stored
Ingesting: meeting_notes.docx   ->   5 chunks stored

Done. Collection 'synapse' in './synapse_db'
```

**From a SQLite table:**

```python
from synapse import ingest_sqlite

ingest_sqlite("./data.db", table="articles")
```

```
Ingesting: articles (120 records)
  -> 87 chunks stored
```

> [!TIP]
> Both sources write to the **same ChromaDB collection** by default. Your agent queries files and database records in a single call.

<details>
<summary>See all <code>ingest()</code> options</summary>

```python
ingest(
    source_dir      = "./docs",             # folder to scan (recursive)
    db_path         = "./synapse_db",       # ChromaDB persistence path
    collection_name = "synapse",            # collection name
    chunk_size      = 1000,                 # target characters per chunk
    overlap         = 200,                  # overlap between consecutive chunks
    min_chunk_size  = 50,                   # discard chunks shorter than this
    embedding_model = "all-MiniLM-L6-v2",  # any SentenceTransformer model name
    verbose         = True,                 # print progress to stdout
)
```

</details>

---

## 🔌 Connecting to an AI agent

synapse handles the **ingestion** half of RAG. The full pattern — ingest once, then query on every user request:

```python
from synapse import ingest

# Step 1 — run once to populate the vector database
ingest("./docs")
```

```python
import chromadb
from chromadb.utils import embedding_functions

# Step 2 — connect your agent to the collection
client = chromadb.PersistentClient(path="./synapse_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
collection = client.get_collection("synapse", embedding_function=ef)

def ask(question: str) -> str:
    results = collection.query(query_texts=[question], n_results=4)
    context = "\n\n".join(results["documents"][0])
    # Step 3 — pass context to your LLM of choice
    return context

print(ask("What is the refund policy?"))
```

> [!IMPORTANT]
> synapse is model-agnostic — it only provides retrieved chunks as context. The same pattern works with Anthropic, OpenAI, Ollama, Mistral, or any other LLM.

> [!NOTE]
> Each chunk carries `source_type` (`"file"` or `"sqlite"`), `source` (absolute path or `db::table`), and `chunk` (index) in its metadata — your agent always knows where an answer came from.

<details>
<summary>Need to run ingest from an async context?</summary>

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

## 🧹 Collection management

Beyond ingestion, synapse exposes three utilities to keep your collection healthy:

### `purge()` — remove stale chunks

When source files are deleted, their chunks remain in ChromaDB. `purge()` cleans them up:

```python
from synapse import purge

purge()  # removes chunks whose source file no longer exists on disk
```

### `reset()` — wipe the collection

Start fresh by deleting the entire collection:

```python
from synapse import reset

reset()
```

### `sources()` — inspect ingested sources

List every source currently stored in the collection (files and databases):

```python
from synapse import sources

for path in sources():
    print(path)
# /home/user/documents/company_policy.pdf
# /home/user/documents/product_faq.txt
# /home/user/data.db::articles
```

All three functions accept the same `db_path` and `collection_name` arguments as `ingest()`.

---

## 📄 Data sources

### Files

| Format | Extensions |
|---|---|
| Plain text | `.txt` `.md` |
| PDF | `.pdf` |
| Word document | `.docx` |
| Spreadsheet | `.csv` |
| JSON | `.json` `.jsonl` |

### SQLite database

Ingest records from any table with `ingest_sqlite()`:

```python
from synapse import ingest_sqlite

ingest_sqlite(
    db_path   = "./data.db",      # path to the SQLite file
    table     = "articles",       # table to ingest
)
```

Each row is serialized to `"col: value | col: value | ..."` and goes through the same chunker and embedder as files. Files and database records end up in the **same ChromaDB collection** — your agent queries both in a single call.

<details>
<summary>► See all <code>ingest_sqlite()</code> options</summary>

```python
ingest_sqlite(
    db_path         = "./data.db",          # path to the SQLite file
    table           = "articles",           # table to ingest
    columns         = None,                 # list of columns to include (None = all)
    id_column       = "id",                 # primary key for stable chunk IDs
    row_template    = None,                 # optional "{title}: {body}" format string
    chroma_path     = "./synapse_db",       # same ChromaDB directory as ingest()
    collection_name = "synapse",            # same collection name
    chunk_size      = 1000,
    overlap         = 200,
    min_chunk_size  = 50,
    embedding_model = "all-MiniLM-L6-v2",
    verbose         = True,
)
```

</details>

> [!NOTE]
> The `source_type` metadata field distinguishes origins: `"file"` for ingested files, `"sqlite"` for database records — so your agent always knows where a result came from.

---

## 🏗️ Architecture

```
synapse/
├── synapse_db/                  ← ChromaDB writes here (auto-created)
│
└── synapse/
    ├── __init__.py              ← public API: ingest, ingest_sqlite, purge, reset, sources
    ├── pipeline.py              ← file ingestion pipeline
    │     ingest()               ·  scan → extract → chunk → embed → upsert
    │     purge()                ·  delete chunks with missing source files
    │     reset()                ·  wipe the entire collection
    │     sources()              ·  list unique ingested source paths
    │
    ├── sqlite_ingester.py       ← SQLite ingestion pipeline
    │     ingest_sqlite()        ·  connect → fetch rows → serialize → chunk → embed → upsert
    │
    ├── extractors.py            ← file extension → raw text string
    │     .txt .md               ·  built-in open()
    │     .pdf                   ·  pypdf
    │     .docx                  ·  python-docx
    │     .csv                   ·  built-in csv
    │     .json .jsonl           ·  built-in json, recursive value flattening
    │
    └── chunker.py               ← raw text → overlapping chunks
          chunk_text()           ·  word-boundary aware sliding window
                                 ·  configurable size, overlap, min_chunk_size
```

Data flow from source to vector store:

```
  ┌─────────────────┐     ┌──────────────────┐
  │  File on disk   │     │  SQLite table    │
  └────────┬────────┘     └────────┬─────────┘
           │ extractors.py          │ sqlite_ingester.py
           │ (text extraction)      │ (row serialization)
           ▼                        ▼
  ┌────────────────────────────────────────┐
  │              Raw text string           │
  └────────────────────┬───────────────────┘
                       │ chunker.py  (word-boundary sliding window)
                       ▼
  ┌──────────────────────────────────────┐
  │  chunk 0  │  chunk 1  │  chunk 2 … │
  └────────────────────┬─────────────────┘
                       │ SentenceTransformer embedding (local)
                       ▼
  ┌────────────────────────────────────────────────────────────┐
  │  ChromaDB  ·  vectors + metadata                          │
  │  { source_type: "file",   source: "/docs/report.pdf" }   │
  │  { source_type: "sqlite", source: "/data.db::articles" } │
  └────────────────────────────────────────────────────────────┘
```

---

## 🗺️ Roadmap

- [x] **7 file formats** — `txt`, `md`, `pdf`, `docx`, `csv`, `json`, `jsonl`
- [x] **Word-boundary chunking** — no mid-word cuts, configurable size, overlap and minimum chunk size
- [x] **Local embeddings** — `sentence-transformers`, no API key, fully offline
- [x] **ChromaDB** — persistent vector store, zero config
- [x] **Idempotent ingestion** — upsert on re-run, never duplicates
- [x] **Collection management** — `purge()`, `reset()`, `sources()`
- [x] **CI/CD** — GitHub Actions pipeline across Python 3.9–3.13
- [x] **SQLite ingestion** — `ingest_sqlite()` to embed table records alongside files
- [ ] **PyPI release** — publish so `pip install synapse` works out of the box
- [ ] **More formats** — `.pptx`, `.xlsx`, `.html`, `.epub`, `.odt`
- [ ] **Incremental ingestion** — skip unchanged files (hash or mtime check) for faster re-runs
- [ ] **File watcher** — `watch()` that monitors `./docs` and auto-ingests on change
- [ ] **Semantic chunking** — split on sentence and paragraph boundaries for better chunk coherence
- [ ] **Pluggable embedders** — OpenAI, Cohere, HuggingFace Inference API as drop-in alternatives
- [ ] **Pluggable vector stores** — Qdrant, FAISS, Weaviate as alternatives to ChromaDB
- [ ] **Document metadata** — extract and store PDF author, creation date, title automatically
- [ ] **Re-ranking** — cross-encoder re-ranking of retrieved chunks before returning context
- [ ] **CLI** — `synapse ingest`, `synapse purge`, `synapse sources` terminal commands
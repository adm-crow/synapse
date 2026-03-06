<div align="center">
  <img src="logo.svg" alt="Synapse" width="120" /><br/><br/>

# ⚡Synapse

[![CI](https://github.com/adm-crow/synapse/actions/workflows/ci.yml/badge.svg)](https://github.com/adm-crow/synapse/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-45%20passing-brightgreen?style=flat-square)](tests/)
[![python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-Apache%202.0-brightgreen?style=flat-square)](LICENSE)
[![pypi](https://img.shields.io/pypi/v/synapse-core?style=flat-square&label=pypi)](https://pypi.org/project/synapse-core/)

</div>

**synapse** is a local-first Python library that turns your files and databases into a searchable vector store — ready for any AI agent to query.

```
Files / SQLite  ──►  Extract  ──►  Chunk  ──►  Embed  ──►  ChromaDB
```

| | Feature | Details |
|---|---|---|
| 📄 | **7 file formats** | `txt`, `md`, `csv`, `pdf`, `docx`, `json`, `jsonl` |
| 🗄️ | **SQLite ingestion** | Embed table records alongside files |
| ✂️ | **Smart chunking** | Word-boundary aware, configurable size & overlap |
| 🧠 | **Local embeddings** | `sentence-transformers` — no API key, fully offline |
| 💾 | **ChromaDB** | Persistent vector store, zero config |
| 🔁 | **Idempotent** | Re-run safely — chunks are upserted, never duplicated |
| 🤖 | **Agent agnostic** | Works with LangChain, LlamaIndex, or any custom agent |

---

## Installation

```bash
pip install synapse-core
```
or
```bash
uv add synapse-core
```

---

## Quick start

**Ingest files:**

```python
from synapse_core import ingest

ingest("./my_documents")
```

```
Ingesting: company_policy.pdf   ->  12 chunks stored
Ingesting: product_faq.txt      ->   8 chunks stored
Ingesting: meeting_notes.docx   ->   5 chunks stored

Done. Collection 'synapse' in './synapse_db'
```

**Ingest a SQLite table:**

```python
from synapse_core import ingest_sqlite

ingest_sqlite("./data.db", table="articles")
```

```
Ingesting: articles (120 records)
  -> 87 chunks stored
```

> [!TIP]
> Both sources write to the **same ChromaDB collection** by default — your agent queries files and database records in a single call.

---

## Connecting to an AI agent

synapse handles ingestion. Connect any agent to the collection it populates:

```python
from synapse_core import ingest
import chromadb
from chromadb.utils import embedding_functions

# Step 1 — ingest once
ingest("./docs")

# Step 2 — connect your agent
client = chromadb.PersistentClient(path="./synapse_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
collection = client.get_collection("synapse", embedding_function=ef)

# Step 3 — query on every request
def ask(question: str) -> str:
    results = collection.query(query_texts=[question], n_results=4)
    return "\n\n".join(results["documents"][0])

print(ask("What is the refund policy?"))
```

> [!IMPORTANT]
> synapse is model-agnostic — pass the returned chunks as context to Anthropic, OpenAI, Ollama, or any other LLM.

> [!NOTE]
> Every chunk carries `source_type` (`"file"` or `"sqlite"`), `source` (absolute path or `db::table`), and `chunk` (index) in its metadata.

---

## Collection management

```python
from synapse_core import purge, reset, sources

purge()     # remove chunks whose source file no longer exists on disk
reset()     # wipe the entire collection
sources()   # list all ingested source paths
```

**`purge()`** — when source files are deleted, their chunks stay in ChromaDB until you purge them.

**`reset()`** — start fresh by deleting the entire collection.

**`sources()`** — inspect what's currently indexed:

```python
for path in sources():
    print(path)
# /home/user/docs/policy.pdf
# /home/user/data.db::articles
```

All three accept the same `db_path` and `collection_name` arguments as `ingest()`.

---

## Options

<details>
<summary><code>ingest()</code></summary>

```python
ingest(
    source_dir      = "./docs",             # folder to scan (recursive)
    db_path         = "./synapse_db",       # ChromaDB persistence path
    collection_name = "synapse",            # collection name
    chunk_size      = 1000,                 # target characters per chunk
    overlap         = 200,                  # overlap between consecutive chunks
    min_chunk_size  = 50,                   # discard chunks shorter than this
    embedding_model = "all-MiniLM-L6-v2",  # any SentenceTransformer model name
    verbose         = True,
)
```

</details>

<details>
<summary><code>ingest_sqlite()</code></summary>

```python
ingest_sqlite(
    db_path         = "./data.db",
    table           = "articles",
    columns         = None,                 # list of columns to include (None = all)
    id_column       = "id",                 # primary key for stable chunk IDs
    row_template    = None,                 # optional "{title}: {body}" format string
    chroma_path     = "./synapse_db",
    collection_name = "synapse",
    chunk_size      = 1000,
    overlap         = 200,
    min_chunk_size  = 50,
    embedding_model = "all-MiniLM-L6-v2",
    verbose         = True,
)
```

Each row is serialized to `"col: value | col: value | ..."` before chunking. Use `row_template` to customise the format.

</details>

---

## Architecture

```
synapse/
├── synapse_db/              ← ChromaDB writes here (auto-created)
└── synapse_core/
    ├── __init__.py          ← public API
    ├── pipeline.py          ← ingest · purge · reset · sources
    ├── sqlite_ingester.py   ← ingest_sqlite
    ├── extractors.py        ← txt · md · pdf · docx · csv · json · jsonl
    └── chunker.py           ← word-boundary sliding window
```

---

## Roadmap

- [x] **7 file formats** — `txt`, `md`, `pdf`, `docx`, `csv`, `json`, `jsonl`
- [x] **Word-boundary chunking** — no mid-word cuts, configurable size, overlap and minimum chunk size
- [x] **Local embeddings** — `sentence-transformers`, no API key, fully offline
- [x] **ChromaDB** — persistent vector store, zero config
- [x] **Idempotent ingestion** — upsert on re-run, never duplicates
- [x] **Collection management** — `purge()`, `reset()`, `sources()`
- [x] **CI/CD** — GitHub Actions pipeline across Python 3.9–3.13
- [x] **SQLite ingestion** — `ingest_sqlite()` to embed table records alongside files
- [ ] **PyPI release** — publish so `pip install synapse-core` works out of the box
- [ ] **More formats** — `.pptx`, `.xlsx`, `.html`, `.epub`, `.odt`
- [ ] **Incremental ingestion** — skip unchanged files (hash or mtime check) for faster re-runs
- [ ] **File watcher** — `watch()` that monitors a folder and auto-ingests on change
- [ ] **Semantic chunking** — split on sentence and paragraph boundaries
- [ ] **Pluggable embedders** — OpenAI, Cohere, HuggingFace Inference API as alternatives
- [ ] **Pluggable vector stores** — Qdrant, FAISS, Weaviate as alternatives to ChromaDB
- [ ] **Document metadata** — extract and store PDF author, creation date, title automatically
- [ ] **Re-ranking** — cross-encoder re-ranking of retrieved chunks
- [ ] **CLI** — `synapse ingest`, `synapse purge`, `synapse sources` terminal commands

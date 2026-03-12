<div align="center">
  <img src="logo.svg" alt="Synapse" width="120" /><br/><br/>

# ⚡Synapse

[![CI](https://github.com/adm-crow/synapse/actions/workflows/ci.yml/badge.svg)](https://github.com/adm-crow/synapse/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-62%20passing-brightgreen?style=flat-square)](tests/)
[![python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-Apache%202.0-brightgreen?style=flat-square)](LICENSE)
[![pypi](https://img.shields.io/pypi/v/synapse-core?style=flat-square&label=pypi)](https://pypi.org/project/synapse-core/)

</div>

**synapse** is a local-first Python library that turns your files and databases into a searchable vector store — ingest once, query semantically, pipe results straight into any AI agent.

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
| 🔍 | **Semantic search** | `query()` returns ranked results with scores and source attribution |
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
2026-03-09 14:32:01 - INFO : Ingesting: company_policy.pdf
2026-03-09 14:32:02 - INFO :   -> 12 chunks stored
2026-03-09 14:32:03 - INFO : Ingesting: product_faq.txt
2026-03-09 14:32:04 - INFO :   -> 8 chunks stored
2026-03-09 14:32:05 - INFO : Ingesting: meeting_notes.docx
2026-03-09 14:32:06 - INFO :   -> 5 chunks stored
2026-03-09 14:32:06 - INFO : Done. Collection 'synapse' in './synapse_db'
```

**Ingest a SQLite table:**

```python
from synapse_core import ingest_sqlite

ingest_sqlite("./data.db", table="articles")
```

```
2026-03-09 14:32:07 - INFO : Ingesting: articles (120 records)
2026-03-09 14:32:09 - INFO :   -> 87 chunks stored
```

> [!TIP]
> Both sources write to the **same ChromaDB collection** by default — your agent queries files and database records in a single call.

---

## Connecting to an AI agent

synapse handles ingestion and retrieval — you wire it to any LLM. Here's a complete example with the **Anthropic SDK**:

```bash
pip install synapse-core anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
import anthropic
from synapse_core import ingest, query

# Step 1 — ingest once
ingest("./docs")

# Step 2 — RAG-powered agent
client = anthropic.Anthropic()

def ask(question: str) -> str:
    chunks = query(question, n_results=4)
    context = "\n\n".join(r["text"] for r in chunks)

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

print(ask("What is the refund policy?"))
```

Each `query()` result is a plain dict — no ChromaDB types leak out:

```python
{
    "text":        "chunk content...",
    "source":      "/abs/path/to/file.txt",
    "source_type": "file",    # "file" or "sqlite"
    "score":       0.91,      # relevance 0–1, higher is better
    "distance":    0.09,      # raw ChromaDB L2 distance
    "chunk":       2,         # index within the source document
}
```

> [!IMPORTANT]
> synapse is model-agnostic — swap `anthropic` for `openai`, `ollama`, or any other SDK without changing a line of synapse code.

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
    incremental     = False,                # skip files whose content hasn't changed
    chunking        = "word",               # "word" or "sentence" (requires nltk)
    verbose         = True,
)
```

**`incremental=True`** — on re-runs, each file's SHA-256 hash is compared against the stored hash. Unchanged files are skipped entirely; changed files are deleted and re-ingested. Zero overhead on the first run.

**`chunking="sentence"`** — splits on paragraph then sentence boundaries using `nltk`. Requires `pip install synapse-core[sentence]`. Falls back to word-boundary splitting for sentences longer than `chunk_size`.

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
    chunking        = "word",               # "word" or "sentence" (requires nltk)
    verbose         = True,
)
```

Each row is serialized to `"col: value | col: value | ..."` before chunking. Use `row_template` to customise the format.

</details>

<details>
<summary><code>query()</code></summary>

```python
query(
    text            = "what is the refund policy?",
    db_path         = "./synapse_db",       # ChromaDB persistence path
    collection_name = "synapse",            # must match the name used at ingest
    n_results       = 5,                    # max number of results to return
    embedding_model = "all-MiniLM-L6-v2",  # must match the model used at ingest
)
```

Returns a list of dicts: `text`, `source`, `source_type`, `score`, `distance`, `chunk`.

</details>

---

## Logging

By default synapse writes colored `INFO` messages to stdout. Use `setup_logging()` to customise the level or add a log file:

```python
import logging
import synapse_core

# Add a persistent log file (plain text, no ANSI codes)
synapse_core.setup_logging(log_file="ingest.log")

# More verbose output
synapse_core.setup_logging(level=logging.DEBUG)

# Silence all synapse_core output
synapse_core.setup_logging(level=logging.CRITICAL)
```

You can also control verbosity per-call with the `verbose` parameter (`True` by default on all functions that produce output).

---

## Architecture

```
synapse/
├── synapse_db/              ← ChromaDB writes here (auto-created)
└── synapse_core/
    ├── __init__.py          ← public API
    ├── pipeline.py          ← ingest · query · purge · reset · sources
    ├── sqlite_ingester.py   ← ingest_sqlite
    ├── extractors.py        ← txt · md · pdf · docx · csv · json · jsonl
    ├── chunker.py           ← word-boundary sliding window
    └── logger.py            ← colored logger · setup_logging()
```

---

## Roadmap

- [x] **7 file formats** — `txt`, `md`, `pdf`, `docx`, `csv`, `json`, `jsonl`
- [x] **Word-boundary chunking** — no mid-word cuts, configurable size, overlap and minimum chunk size
- [x] **Local embeddings** — `sentence-transformers`, no API key, fully offline
- [x] **ChromaDB** — persistent vector store, zero config
- [x] **Idempotent ingestion** — upsert on re-run, never duplicates
- [x] **Collection management** — `purge()`, `reset()`, `sources()`
- [x] **CI/CD** — GitHub Actions pipeline across Python 3.11–3.13
- [x] **SQLite ingestion** — `ingest_sqlite()` to embed table records alongside files
- [x] **Semantic search** — `query()` returns ranked results with relevance scores and source attribution
- [x] **Structured logging** — colored output, configurable level, optional file output via `setup_logging()`
- [x] **PyPI release** — `pip install synapse-core`
- [ ] **More formats** — `.pptx`, `.xlsx`, `.html`, `.epub`, `.odt`
- [x] **Incremental ingestion** — skip unchanged files (SHA-256 hash check) for faster re-runs
- [ ] **File watcher** — `watch()` that monitors a folder and auto-ingests on change
- [x] **Semantic chunking** — split on sentence and paragraph boundaries via `chunking="sentence"`
- [ ] **Pluggable embedders** — OpenAI, Cohere, HuggingFace Inference API as alternatives
- [ ] **Pluggable vector stores** — Qdrant, FAISS, Weaviate as alternatives to ChromaDB
- [ ] **Document metadata** — extract and store PDF author, creation date, title automatically
- [ ] **Re-ranking** — cross-encoder re-ranking of retrieved chunks
- [ ] **CLI** — `synapse ingest`, `synapse purge`, `synapse sources` terminal commands

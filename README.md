<div align="center">
  <img src="logo.svg" alt="Synapse" width="140" /><br/><br/>

  <h1>⚡ synapse-core</h1>

  <p><strong>Local-first RAG library for Python — ingest files, query semantically, feed any AI agent.</strong></p>

  [![CI](https://github.com/adm-crow/synapse/actions/workflows/ci.yml/badge.svg)](https://github.com/adm-crow/synapse/actions/workflows/ci.yml)
  [![tests](https://img.shields.io/badge/tests-107%20passing-brightgreen?style=flat-square)](tests/)
  [![PyPI](https://img.shields.io/pypi/v/synapse-core?style=flat-square&color=blue)](https://pypi.org/project/synapse-core/)
  [![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](LICENSE)
  [![Downloads](https://img.shields.io/pypi/dm/synapse-core?style=flat-square&color=orange)](https://pypi.org/project/synapse-core/)

</div>

---

## What is synapse?

**synapse** turns your local files and SQLite databases into a searchable vector store in a few lines of code. No cloud, no API key, no infrastructure — everything runs on your machine.

```
Files / SQLite  ──►  Extract  ──►  Chunk  ──►  Embed  ──►  ChromaDB  ──►  Your AI Agent
```

---

## Features

| | Feature | Details |
|:---:|:---|:---|
| 📄 | **12 file formats** | `txt` `md` `csv` `pdf` `docx` `json` `jsonl` `html` `pptx` `xlsx` `epub` `odt` |
| 🗄️ | **SQLite ingestion** | Embed table records alongside files in the same collection |
| ✂️ | **Smart chunking** | Word-boundary and sentence-aware, configurable size & overlap |
| 🧠 | **Local embeddings** | `sentence-transformers` — no API key, fully offline |
| 💾 | **ChromaDB** | Persistent vector store, zero config |
| 🔁 | **Incremental ingestion** | SHA-256 hash check — skip unchanged files on re-runs |
| 🔍 | **Semantic search** | Ranked results with scores, source path, and document metadata |
| 📋 | **Document metadata** | Auto-extract title, author, creation date from PDF/DOCX/HTML/PPTX |
| 🖥️ | **CLI** | `synapse ingest`, `query`, `sources`, `purge`, `reset` |
| 🤖 | **Agent-agnostic** | Works with Anthropic, OpenAI, Ollama, LangChain — anything |

---

## Installation

```bash
pip install synapse-core
# or
uv add synapse-core
```

**Extra file formats** — `.html`, `.pptx`, `.xlsx`, `.epub`, `.odt`:
```bash
pip install synapse-core[formats]
```

**Sentence-aware chunking** (`chunking="sentence"`):
```bash
pip install synapse-core[sentence]
```

**Everything at once:**
```bash
pip install synapse-core[formats,sentence]
```

---

## Quick start

### Ingest files

```python
from synapse_core import ingest

ingest("./my_documents")
```

```
2026-03-09 14:32:01 - INFO : Ingesting: company_policy.pdf
2026-03-09 14:32:02 - INFO :   -> 12 chunks stored
2026-03-09 14:32:03 - INFO : Ingesting: product_faq.txt
2026-03-09 14:32:04 - INFO :   -> 8 chunks stored
2026-03-09 14:32:06 - INFO : Done. Collection 'synapse' in './synapse_db'
```

### Ingest a SQLite table

```python
from synapse_core import ingest_sqlite

ingest_sqlite("./data.db", table="articles")
```

> [!TIP]
> Both sources write to the **same ChromaDB collection** by default — your agent queries files and database records in a single call.

### Query semantically

```python
from synapse_core import query

results = query("what is the refund policy?", n_results=4)

for r in results:
    print(f"[{r['score']:.2f}] {r['source']}")
    print(r['text'])
```

---

## CLI

After installation a `synapse` command is available in your terminal.

### Commands

```bash
# Ingest a folder
synapse ingest ./docs

# Re-ingest, skipping unchanged files
synapse ingest ./docs --incremental

# Ingest with sentence-aware chunking
synapse ingest ./docs --chunking sentence

# Ingest a SQLite table
synapse ingest-sqlite ./data.db --table articles

# Semantic search (raw chunks)
synapse query "what is the refund policy?"

# AI-powered answer — set your API key first, then query

# Anthropic — macOS/Linux
export ANTHROPIC_API_KEY="sk-ant-..."
# Anthropic — Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."

synapse query "what is the refund policy?" --ai
synapse query "what is the refund policy?" --ai --provider anthropic --model claude-sonnet-4-5

# OpenAI — macOS/Linux
export OPENAI_API_KEY="sk-..."
# OpenAI — Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

synapse query "what is the refund policy?" --ai --provider openai --model gpt-4o

# Ollama (local, no key needed — just start the server)
ollama serve
synapse query "what is the refund policy?" --ai --provider ollama --model mistral

# List all indexed sources
synapse sources

# Remove chunks from deleted source files
synapse purge

# Wipe the entire collection
synapse reset --yes
```

### Global options

Every command accepts these options to target a specific store:

| Option | Default | Description |
|:---|:---|:---|
| `--db PATH` | `./synapse_db` | ChromaDB persistence path |
| `--collection NAME` | `synapse` | Collection name |

```bash
# Example: use a custom store
synapse query "auth flow" --db ./my_store --collection docs
```

> Run `synapse --help` or `synapse <command> --help` for the full option list.

---

## Connecting to an AI agent

synapse handles ingestion and retrieval — you wire it to any LLM. Here's a complete example with the **Anthropic SDK**:

```python
import anthropic
from synapse_core import ingest, query

# 1 — ingest your documents (once)
ingest("./docs")

# 2 — build a RAG-powered assistant
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

def ask(question: str) -> str:
    chunks = query(question, n_results=4)
    context = "\n\n".join(r["text"] for r in chunks)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=(
            "You are a helpful assistant. "
            "Answer using ONLY the context below. "
            "If the answer is not in the context, say so.\n\n"
            f"CONTEXT:\n{context}"
        ),
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text

print(ask("What is the refund policy?"))
```

> [!IMPORTANT]
> synapse is model-agnostic — swap `anthropic` for `openai`, `ollama`, or any other SDK without changing a single line of synapse code.

### Query result shape

Each `query()` result is a plain dict — no ChromaDB types leak out:

```python
{
    "text":        "chunk content...",
    "source":      "/abs/path/to/file.txt",
    "source_type": "file",              # "file" or "sqlite"
    "score":       0.91,                # relevance 0–1, higher is better
    "distance":    0.09,                # raw ChromaDB L2 distance
    "chunk":       2,                   # index within the source document
    "doc_title":   "Company Policy",   # from PDF/DOCX/HTML/PPTX metadata
    "doc_author":  "Jane Doe",          # document author, "" if unavailable
    "doc_created": "2024-01-15T...",   # ISO-8601 creation date, "" if unavailable
}
```

---

## Collection management

```python
from synapse_core import purge, reset, sources

sources()   # list all ingested source paths
purge()     # remove chunks whose source file no longer exists
reset()     # wipe the entire collection and start fresh
```

**`sources()`** — inspect what's indexed:
```python
for path in sources():
    print(path)
# /home/user/docs/policy.pdf
# /home/user/data.db::articles
```

**`purge()`** — when source files are deleted, their chunks stay in ChromaDB until you purge.

**`reset()`** — deletes the entire collection. Irreversible. Use with care.

All three accept the same `db_path` and `collection_name` arguments as `ingest()`.

---

## API reference

<details>
<summary><strong><code>ingest()</code></strong></summary>

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
    chunking        = "word",               # "word" or "sentence" (requires [sentence])
    verbose         = True,
)
```

**`incremental=True`** — compares each file's SHA-256 hash against the stored hash. Unchanged files are skipped; changed files are re-ingested from scratch. Zero overhead on first run.

**`chunking="sentence"`** — splits on paragraph then sentence boundaries via `nltk`. Falls back to word-boundary splitting for sentences longer than `chunk_size`.

</details>

<details>
<summary><strong><code>ingest_sqlite()</code></strong></summary>

```python
ingest_sqlite(
    db_path         = "./data.db",
    table           = "articles",
    columns         = None,                 # list of columns to embed (None = all)
    id_column       = "id",                 # primary key for stable chunk IDs
    row_template    = None,                 # optional "{title}: {body}" format string
    chroma_path     = "./synapse_db",
    collection_name = "synapse",
    chunk_size      = 1000,
    overlap         = 200,
    min_chunk_size  = 50,
    embedding_model = "all-MiniLM-L6-v2",
    chunking        = "word",
    verbose         = True,
)
```

Each row is serialized to `"col: value | col: value | ..."` before chunking. Use `row_template` to customise the format.

</details>

<details>
<summary><strong><code>query()</code></strong></summary>

```python
query(
    text            = "what is the refund policy?",
    db_path         = "./synapse_db",       # ChromaDB persistence path
    collection_name = "synapse",            # must match the name used at ingest
    n_results       = 5,                    # max number of results to return
    embedding_model = "all-MiniLM-L6-v2",  # must match the model used at ingest
)
```

Returns a list of dicts: `text`, `source`, `source_type`, `score`, `distance`, `chunk`, `doc_title`, `doc_author`, `doc_created`.

</details>

---

## Logging

By default synapse writes colored `INFO` messages to stdout. Use `setup_logging()` to customise:

```python
import logging
import synapse_core

synapse_core.setup_logging(log_file="ingest.log")       # add a persistent log file
synapse_core.setup_logging(level=logging.DEBUG)          # more verbose
synapse_core.setup_logging(level=logging.CRITICAL)       # silence all output
```

You can also pass `verbose=False` to any `ingest()` or `query()` call to suppress output for that call only.

---

## Architecture

```
synapse/
├── synapse_db/              ← ChromaDB writes here (auto-created)
└── synapse_core/
    ├── __init__.py          ← public API
    ├── cli.py               ← synapse ingest · ingest-sqlite · query · sources · purge · reset
    ├── pipeline.py          ← ingest() · query() · purge() · reset() · sources()
    ├── sqlite_ingester.py   ← ingest_sqlite()
    ├── extractors.py        ← 12 formats + document metadata extraction
    ├── chunker.py           ← word-boundary & sentence-aware chunking
    └── logger.py            ← colored logger · setup_logging()
```

---

## Roadmap

- [x] 7 file formats — `txt`, `md`, `pdf`, `docx`, `csv`, `json`, `jsonl`
- [x] Word-boundary chunking — no mid-word cuts, configurable size, overlap and minimum chunk size
- [x] Local embeddings — `sentence-transformers`, no API key, fully offline
- [x] ChromaDB — persistent vector store, zero config
- [x] Idempotent ingestion — upsert on re-run, never duplicates
- [x] Collection management — `purge()`, `reset()`, `sources()`
- [x] CI/CD — GitHub Actions pipeline across Python 3.11–3.13
- [x] SQLite ingestion — `ingest_sqlite()` to embed table records alongside files
- [x] Semantic search — ranked results with relevance scores and source attribution
- [x] Structured logging — colored output, configurable level, optional file output
- [x] PyPI release — `pip install synapse-core`
- [x] Incremental ingestion — skip unchanged files (SHA-256 hash) for faster re-runs
- [x] Sentence chunking — split on sentence and paragraph boundaries via `chunking="sentence"`
- [x] More formats — `.html`, `.pptx`, `.xlsx`, `.epub`, `.odt` via `[formats]` extra
- [x] Document metadata — auto-extract and store title, author, creation date
- [x] CLI — `synapse ingest`, `ingest-sqlite`, `query`, `purge`, `reset`, `sources`
- [ ] File watcher — `watch()` that monitors a folder and auto-ingests on change
- [ ] Pluggable embedders — OpenAI, Cohere, HuggingFace Inference API as alternatives
- [ ] Pluggable vector stores — Qdrant, FAISS, Weaviate as alternatives to ChromaDB
- [ ] Re-ranking — cross-encoder re-ranking of retrieved chunks

---

<div align="center">
  <sub>Built with ❤️ · <a href="https://pypi.org/project/synapse-core/">PyPI</a> · <a href="LICENSE">Apache 2.0</a></sub>
</div>

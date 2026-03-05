<div align="center">

<img src="logo.svg" alt="Synapse logo" width="160" /><br/><br/>

# synapse

**Turn your files into answers.**

*Drop documents. Run one function. Let any AI agent query your knowledge.*

<br/>

[![Python](https://img.shields.io/badge/python-3.9%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/vector--db-ChromaDB-FF6719?style=for-the-badge)](https://www.trychroma.com/)
[![sentence-transformers](https://img.shields.io/badge/embeddings-sentence--transformers-4B8BBE?style=for-the-badge)](https://www.sbert.net/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-21%20passed-22c55e?style=for-the-badge&logo=pytest&logoColor=white)](tests/)

<br/>

</div>

---

## What is Synapse?

Synapse is a **local-first RAG ingestion pipeline** packaged as a minimal Python library.

You drop files into a folder. Synapse extracts the text, splits it into chunks, converts them to vectors using a local embedding model, and stores everything in ChromaDB — on your machine, with zero cloud dependency.

Your AI agent then queries that collection to retrieve relevant context for any question.

```
  ./docs/                          ./synapse_db/
  ├── contract.pdf                 ╔══════════════════════╗
  ├── faq.txt       ─ ingest() ─►  ║  ChromaDB collection ║
  └── notes.docx                   ║  vectors + metadata  ║
                                   ╚══════════════════════╝
                                            │
                                            ▼
                                    🤖 Your AI agent
```

---

## ✨ Highlights

| | |
|---|---|
| 🏠 **Local first** | No API keys. No cloud. Everything runs on your machine. |
| ⚡ **One function** | `ingest()` is all you need to go from files to vectors. |
| 🔁 **Idempotent** | Re-run safely — chunks are updated, never duplicated. |
| 🤖 **Agent agnostic** | Works with LangChain, LlamaIndex, or any custom agent. |
| 📁 **Recursive scan** | Automatically picks up files in subdirectories. |
| 🔌 **Extensible** | Add new file types or swap the embedding model easily. |

---

## 📄 Supported formats

| Format | Extension |
|---|---|
| Plain text | `.txt` `.md` |
| PDF | `.pdf` |
| Word document | `.docx` `.doc` |
| Spreadsheet | `.csv` |

---

## 🚀 Installation

```bash
pip install -e .
```

Dependencies (`chromadb`, `sentence-transformers`, `pypdf`, `python-docx`) are installed automatically.

---

## 🎯 Quick start

### Step 1 — Drop your files

```
./docs/
├── company_policy.pdf
├── product_faq.txt
└── meeting_notes.docx
```

### Step 2 — Ingest

```python
from synapse import ingest

ingest()
```

```
Ingesting: company_policy.pdf  ->  12 chunks stored
Ingesting: product_faq.txt     ->   8 chunks stored
Ingesting: meeting_notes.docx  ->   5 chunks stored

Done. Collection 'synapse' in './synapse_db'
```

### Step 3 — Query from your agent

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./synapse_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
collection = client.get_collection("synapse", embedding_function=ef)

results = collection.query(
    query_texts=["What is the refund policy?"],
    n_results=5,
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"[{meta['source']}]\n{doc[:300]}\n")
```

> See [`examples/quickstart.py`](examples/quickstart.py) for the full working example.

---

## ⚙️ API reference

### `ingest()`

```python
from synapse import ingest

ingest(
    source_dir      = "./docs",             # 📁 folder to scan
    db_path         = "./synapse_db",       # 💾 ChromaDB persistence path
    collection_name = "synapse",            # 🏷️  collection name
    chunk_size      = 1000,                 # ✂️  characters per chunk
    overlap         = 200,                  # 🔗 overlap between chunks
    embedding_model = "all-MiniLM-L6-v2",  # 🧠 SentenceTransformer model
    verbose         = True,                 # 🖨️  print progress
)
```

All parameters are optional — `ingest()` works out of the box with zero configuration.

---

## 🏗️ How it works

```
┌──────────────┐
│  File on disk │
└──────┬───────┘
       │  extract()
       ▼
┌──────────────┐
│   Raw text   │
└──────┬───────┘
       │  chunk_text()
       ▼
┌──────────────────────────────────────┐
│  chunk 1  │  chunk 2  │  chunk 3 ... │   (1000 chars, 200 overlap)
└──────┬───────────────────────────────┘
       │  SentenceTransformer embed
       ▼
┌────────────────────────┐
│  ChromaDB (local disk) │  ← upsert(documents, embeddings, metadata)
└────────────────────────┘
```

Each chunk is stored with its **source file path** and **chunk index** as metadata, so your agent always knows where an answer came from.

---

## 🗂️ Project structure

```
synapse/
│
├── 📁 docs/                    ← drop your files here
│
├── 📁 synapse/
│   ├── __init__.py             ← public API: ingest()
│   ├── pipeline.py             ← orchestrates the full pipeline
│   ├── extractors.py           ← file type → raw text
│   └── chunker.py              ← raw text → overlapping chunks
│
├── 📁 examples/
│   └── quickstart.py           ← minimal agent query example
│
└── 📁 tests/
    ├── test_pipeline.py
    ├── test_extractors.py
    └── test_chunker.py
```

---

## 🧪 Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

```
tests/test_chunker.py::test_empty_text_returns_empty_list     PASSED
tests/test_chunker.py::test_whitespace_is_normalized          PASSED
tests/test_extractors.py::test_extract_txt                    PASSED
tests/test_extractors.py::test_extract_pdf                    PASSED
tests/test_pipeline.py::test_ingest_is_idempotent             PASSED
...
21 passed in 22s
```

---

## 📝 License

MIT — free to use, modify, and distribute.

---

<div align="center">
  <sub>Built with ❤️ — local AI should be simple.</sub>
</div>

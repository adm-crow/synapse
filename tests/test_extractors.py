import json
import tempfile
from pathlib import Path

import pytest

from synapse_core.extractors import extract, is_supported


# --- helpers ---

def write_temp(suffix: str, content: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w", encoding="utf-8")
    f.write(content)
    f.close()
    return Path(f.name)


# --- txt / md ---

def test_extract_txt():
    path = write_temp(".txt", "Hello from txt")
    assert extract(path) == "Hello from txt"


def test_extract_md():
    path = write_temp(".md", "# Title\nSome content")
    assert "Title" in extract(path)


# --- csv ---

def test_extract_csv():
    path = write_temp(".csv", "name,age\nAlice,30\nBob,25")
    result = extract(path)
    assert "Alice" in result
    assert "Bob" in result


# --- pdf ---

def test_extract_pdf():
    pytest.importorskip("pypdf")
    import io
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(buf.getvalue())
        path = Path(f.name)

    result = extract(path)
    assert isinstance(result, str)


# --- docx ---

def test_extract_docx():
    pytest.importorskip("docx")
    from docx import Document

    doc = Document()
    doc.add_paragraph("Hello from docx")
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        doc.save(f.name)
        path = Path(f.name)

    assert "Hello from docx" in extract(path)


# --- json ---

def test_extract_json():
    data = {"title": "RAG guide", "body": "Retrieval Augmented Generation", "year": 2024}
    path = write_temp(".json", json.dumps(data))
    result = extract(path)
    assert "RAG guide" in result
    assert "Retrieval Augmented Generation" in result


def test_extract_jsonl():
    lines = [
        json.dumps({"text": "First document"}),
        json.dumps({"text": "Second document"}),
    ]
    path = write_temp(".jsonl", "\n".join(lines))
    result = extract(path)
    assert "First document" in result
    assert "Second document" in result


# --- unsupported ---

def test_unsupported_extension_raises():
    path = write_temp(".xyz", "data")
    with pytest.raises(ValueError, match="Unsupported"):
        extract(path)


def test_doc_is_not_supported():
    # Legacy .doc (binary Word) is not supported — python-docx only handles .docx
    assert not is_supported(Path("file.doc"))


def test_is_supported():
    assert is_supported(Path("file.txt"))
    assert is_supported(Path("file.PDF"))  # case-insensitive
    assert is_supported(Path("file.json"))
    assert is_supported(Path("file.jsonl"))
    assert not is_supported(Path("file.mp3"))

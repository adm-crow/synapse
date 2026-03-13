import io
import json
import tempfile
from pathlib import Path

import pytest

from synapse_core.extractors import extract, is_supported


# --- helpers ---

def write_temp(tmp_path: Path, suffix: str, content: str) -> Path:
    path = tmp_path / f"file{suffix}"
    path.write_text(content, encoding="utf-8")
    return path


# --- txt / md ---

def test_extract_txt(tmp_path):
    path = write_temp(tmp_path, ".txt", "Hello from txt")
    assert extract(path) == "Hello from txt"


def test_extract_md(tmp_path):
    path = write_temp(tmp_path, ".md", "# Title\nSome content")
    assert "Title" in extract(path)


# --- csv ---

def test_extract_csv(tmp_path):
    path = write_temp(tmp_path, ".csv", "name,age\nAlice,30\nBob,25")
    result = extract(path)
    assert "Alice" in result
    assert "Bob" in result


# --- pdf ---

def test_extract_pdf(tmp_path):
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)

    path = tmp_path / "file.pdf"
    path.write_bytes(buf.getvalue())

    result = extract(path)
    assert isinstance(result, str)


# --- docx ---

def test_extract_docx(tmp_path):
    pytest.importorskip("docx")
    from docx import Document

    doc = Document()
    doc.add_paragraph("Hello from docx")
    path = tmp_path / "file.docx"
    doc.save(str(path))

    assert "Hello from docx" in extract(path)


# --- json ---

def test_extract_json(tmp_path):
    data = {"title": "RAG guide", "body": "Retrieval Augmented Generation", "year": 2024}
    path = write_temp(tmp_path, ".json", json.dumps(data))
    result = extract(path)
    assert "RAG guide" in result
    assert "Retrieval Augmented Generation" in result


def test_extract_jsonl(tmp_path):
    lines = [
        json.dumps({"text": "First document"}),
        json.dumps({"text": "Second document"}),
    ]
    path = write_temp(tmp_path, ".jsonl", "\n".join(lines))
    result = extract(path)
    assert "First document" in result
    assert "Second document" in result


# --- unsupported ---

def test_unsupported_extension_raises(tmp_path):
    path = write_temp(tmp_path, ".xyz", "data")
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

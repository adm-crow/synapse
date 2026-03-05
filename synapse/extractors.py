import csv
import json
from pathlib import Path


def extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_csv(path: Path) -> str:
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        return "\n".join(", ".join(row) for row in reader)


def _flatten_json(obj) -> str:
    """Recursively extract all string values from a parsed JSON object."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(_flatten_json(v) for v in obj.values() if v is not None)
    if isinstance(obj, list):
        return " ".join(_flatten_json(item) for item in obj)
    return str(obj) if obj is not None else ""


def extract_json(path: Path) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    return _flatten_json(data)


def extract_jsonl(path: Path) -> str:
    parts = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts.append(_flatten_json(json.loads(line)))
            except json.JSONDecodeError:
                pass
    return "\n".join(parts)


EXTRACTORS = {
    ".txt": extract_txt,
    ".md": extract_txt,
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".csv": extract_csv,
    ".json": extract_json,
    ".jsonl": extract_jsonl,
}


def extract(path: Path) -> str:
    suffix = path.suffix.lower()
    extractor = EXTRACTORS.get(suffix)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    return extractor(path)


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in EXTRACTORS

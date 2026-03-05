import csv
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


EXTRACTORS = {
    ".txt": extract_txt,
    ".md": extract_txt,
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".doc": extract_docx,  # python-docx can handle some .doc files
    ".csv": extract_csv,
}


def extract(path: Path) -> str:
    suffix = path.suffix.lower()
    extractor = EXTRACTORS.get(suffix)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    return extractor(path)


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in EXTRACTORS

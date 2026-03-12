import re
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    min_chunk_size: int = 50,
    mode: str = "word",
) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text:           Input text to split.
        chunk_size:     Target character count per chunk.
        overlap:        Character overlap between consecutive chunks.
        min_chunk_size: Chunks shorter than this are discarded.
        mode:           "word" (default) snaps to word boundaries;
                        "sentence" splits on sentence/paragraph boundaries
                        using nltk (requires: pip install nltk).
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    if mode not in ("word", "sentence"):
        raise ValueError(f"mode must be 'word' or 'sentence', got '{mode}'")

    if mode == "sentence":
        return _chunk_by_sentences(text, chunk_size, overlap, min_chunk_size)

    # --- word mode ---
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Snap to the nearest word boundary to avoid cutting mid-word
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)

        advance = end - start - overlap
        if advance <= 0:
            advance = max(1, chunk_size - overlap)
        start += advance

    return chunks


def _chunk_by_sentences(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> List[str]:
    """Sentence-aware chunking: split on paragraph → sentence boundaries.

    Overlap is character-based: sentences from the tail of each chunk are
    carried into the next chunk until the overlap budget is exhausted.
    Single sentences that exceed chunk_size fall back to word chunking.
    """
    try:
        import nltk
    except ImportError:
        raise ImportError(
            "sentence chunking requires nltk: pip install nltk\n"
            "Then run: python -c \"import nltk; nltk.download('punkt_tab')\""
        )

    # Ensure punkt tokenizer data is available
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    # Split into paragraphs, then tokenize each paragraph into sentences
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences: List[str] = []
    for para in paragraphs:
        sentences.extend(nltk.sent_tokenize(para))

    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Single sentence exceeds chunk_size — fall back to word chunking
        if len(sent) > chunk_size:
            if current:
                chunk = " ".join(current)
                if len(chunk) >= min_chunk_size:
                    chunks.append(chunk)
                current, current_len = [], 0
            chunks.extend(
                chunk_text(sent, chunk_size=chunk_size, overlap=overlap,
                           min_chunk_size=min_chunk_size, mode="word")
            )
            continue

        # Adding this sentence would exceed chunk_size — emit and start new
        if current and current_len + len(sent) + 1 > chunk_size:
            chunk = " ".join(current)
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)

            # Carry trailing sentences into next chunk up to `overlap` chars
            overlap_sents: List[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) + 1 <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break
            current = overlap_sents
            current_len = overlap_len

        current.append(sent)
        current_len += len(sent) + 1

    # Emit the last chunk
    if current:
        chunk = " ".join(current)
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)

    return chunks

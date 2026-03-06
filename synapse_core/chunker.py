from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    min_chunk_size: int = 50,
) -> List[str]:
    """Split text into overlapping chunks, snapping to word boundaries.

    Args:
        text:           Input text to split.
        chunk_size:     Target character count per chunk.
        overlap:        Character overlap between consecutive chunks.
        min_chunk_size: Chunks shorter than this are discarded.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

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

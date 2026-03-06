import pytest

from synapse_core.chunker import chunk_text


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []


def test_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\t  ") == []


def test_short_text_returns_single_chunk():
    chunks = chunk_text("Hello world", chunk_size=500, min_chunk_size=1)
    assert chunks == ["Hello world"]


def test_long_text_is_split():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1


def test_overlap_is_applied():
    text = "a" * 1000
    chunks_no_overlap = chunk_text(text, chunk_size=500, overlap=0)
    chunks_with_overlap = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)


def test_chunks_cover_full_text():
    text = "word " * 200
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert chunks[0].startswith("word")
    assert "word" in "".join(chunks)


def test_whitespace_is_normalized():
    text = "hello   \n\n  world"
    chunks = chunk_text(text, chunk_size=500, min_chunk_size=1)
    assert chunks == ["hello world"]


def test_word_boundary_respected():
    # Chunks should not end mid-word
    text = "hello world " * 50  # 600 chars of alternating words
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    for chunk in chunks[:-1]:  # last chunk naturally ends at text boundary
        last_word = chunk.split()[-1] if chunk.split() else ""
        assert last_word in ("hello", "world"), (
            f"Chunk ends with partial word: '{last_word}'"
        )


def test_overlap_gte_chunk_size_raises():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_min_chunk_size_filters_tiny_chunks():
    # Produce a tiny trailing chunk (1 char) that should be filtered out
    text = "a" * 100 + " b"  # after normalization: 102 chars, last chunk = "b"
    chunks = chunk_text(text, chunk_size=100, overlap=0, min_chunk_size=10)
    for chunk in chunks:
        assert len(chunk) >= 10

"""Unit tests for corpus/chunker.py."""

import pytest

from ragbench.corpus.chunker import chunk_documents, chunk_text


class TestChunkText:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\t  ") == []

    def test_single_chunk_when_text_fits(self):
        text = " ".join(["word"] * 10)
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_into_correct_number_of_chunks(self):
        # 100 words, chunk_size=40, overlap=10, stride=30 → ceil((100-40)/30)+1 = 3 full + 1 tail
        text = " ".join([f"w{i}" for i in range(100)])
        chunks = chunk_text(text, chunk_size=40, chunk_overlap=10)
        assert len(chunks) >= 3

    def test_overlap_preserved(self):
        words = [f"w{i}" for i in range(20)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=10, chunk_overlap=3)
        # Last 3 words of chunk[0] should appear as first 3 words of chunk[1]
        c0_words = chunks[0].split()
        c1_words = chunks[1].split()
        assert c0_words[-3:] == c1_words[:3]

    def test_last_chunk_contains_final_words(self):
        words = [f"w{i}" for i in range(25)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
        all_words_in_last_chunk = set(chunks[-1].split())
        assert "w24" in all_words_in_last_chunk

    def test_deterministic(self):
        text = " ".join([f"tok{i}" for i in range(200)])
        assert chunk_text(text) == chunk_text(text)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_text("hello world", chunk_size=5, chunk_overlap=5)

    def test_overlap_larger_than_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("a b c d", chunk_size=3, chunk_overlap=4)

    def test_chunk_size_respected(self):
        text = " ".join([f"w{i}" for i in range(100)])
        chunk_size = 15
        for chunk in chunk_text(text, chunk_size=chunk_size, chunk_overlap=3):
            assert len(chunk.split()) <= chunk_size

    def test_exact_size_text(self):
        text = " ".join([f"w{i}" for i in range(40)])
        chunks = chunk_text(text, chunk_size=40, chunk_overlap=5)
        assert len(chunks) == 1

    def test_one_word_input(self):
        chunks = chunk_text("hello", chunk_size=10, chunk_overlap=2)
        assert chunks == ["hello"]


class TestChunkDocuments:
    def test_adds_chunk_metadata(self):
        docs = [{"id": "doc1", "text": " ".join([f"w{i}" for i in range(50)])}]
        chunks = chunk_documents(docs, chunk_size=20, chunk_overlap=5)
        assert all("chunk_id" in c for c in chunks)
        assert all("chunk_index" in c for c in chunks)

    def test_chunk_id_format(self):
        docs = [{"id": "doc1", "text": " ".join(["x"] * 30)}]
        chunks = chunk_documents(docs, chunk_size=20, chunk_overlap=5)
        assert chunks[0]["chunk_id"].startswith("doc1_c")

    def test_preserves_source_metadata(self):
        docs = [{"id": "doc1", "text": "hello world", "ticker": "AAPL"}]
        chunks = chunk_documents(docs, chunk_size=10, chunk_overlap=1)
        for c in chunks:
            assert c["ticker"] == "AAPL"

    def test_empty_documents_list(self):
        assert chunk_documents([]) == []

    def test_multiple_documents(self):
        docs = [
            {"id": "d1", "text": " ".join(["a"] * 50)},
            {"id": "d2", "text": " ".join(["b"] * 50)},
        ]
        chunks = chunk_documents(docs, chunk_size=20, chunk_overlap=5)
        ids = {c["id"] for c in chunks}
        assert "d1" in ids and "d2" in ids

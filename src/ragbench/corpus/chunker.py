"""Fixed-size word-count chunker with overlap.

Parameters are set in configs/corpus.yaml and recorded in DECISIONS.md.
The algorithm is deterministic: identical input always produces identical chunks.
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 40) -> list[str]:
    """Split *text* into overlapping word-count chunks.

    Args:
        text: Input text (any whitespace-normalised string).
        chunk_size: Maximum number of words per chunk.
        chunk_overlap: Number of words shared between consecutive chunks.

    Returns:
        List of chunk strings in document order. Empty input returns [].
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    stride = chunk_size - chunk_overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride

    return chunks


def chunk_documents(
    documents: list[dict[str, str]],
    chunk_size: int = 400,
    chunk_overlap: int = 40,
) -> list[dict[str, str]]:
    """Chunk a list of document dicts, preserving source metadata.

    Each input document must have at least ``"id"`` and ``"text"`` keys.
    Output chunk dicts add ``"chunk_index"`` and ``"chunk_id"`` (``{doc_id}_c{i}``).
    """
    result: list[dict[str, str]] = []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["text"], chunk_size, chunk_overlap)):
            result.append(
                {
                    **{k: v for k, v in doc.items() if k != "text"},
                    "text": chunk,
                    "chunk_index": str(i),
                    "chunk_id": f"{doc['id']}_c{i}",
                }
            )
    return result

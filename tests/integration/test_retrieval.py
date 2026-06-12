"""Retrieval integration tests: index known chunks, verify correct retrieval.

Uses Qdrant in-memory mode — no Docker required, but grpcio + protobuf must be compatible.
Run with: pytest tests/integration/ -m integration

The planted-query test is the ROADMAP Phase 2 acceptance criterion:
  "Retrieval unit tests on fixtures (known chunk must be retrieved for a planted query)."
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration

qdrant_client_mod = pytest.importorskip(  # noqa: E402
    "qdrant_client",
    reason="qdrant_client import failed (likely grpcio/protobuf mismatch — see DECISIONS.md)",
)
QdrantClient = qdrant_client_mod.QdrantClient

from ragbench.retrieval.indexer import QdrantIndexer, _chunk_id_to_uint64  # noqa: E402
from ragbench.retrieval.retriever import Retriever  # noqa: E402

COLLECTION = "test_ragbench"

# Three fixture chunks with distinct topics
FIXTURE_CHUNKS = [
    {
        "chunk_id": "aapl_c0",
        "id": "aapl",
        "text": "Apple reported net income of 99.8 billion dollars for fiscal year 2022 driven by iPhone sales",
        "source_path": "tests/fixtures",
    },
    {
        "chunk_id": "jpm_c0",
        "id": "jpm",
        "text": "JPMorgan Chase reported return on equity of 13 percent for full year 2022 driven by consumer banking",
        "source_path": "tests/fixtures",
    },
    {
        "chunk_id": "xom_c0",
        "id": "xom",
        "text": "ExxonMobil reported record earnings of 55 billion dollars in 2022 due to high oil prices",
        "source_path": "tests/fixtures",
    },
]

# Dimensionality for stub embeddings (must match what we create the collection with)
DIM = 8


def _stub_embeddings(chunks: list[dict]) -> np.ndarray:
    """Assign each chunk a fixed orthogonal unit vector (reproducible, no model needed)."""
    vecs = np.zeros((len(chunks), DIM), dtype=np.float32)
    for i in range(len(chunks)):
        vecs[i, i % DIM] = 1.0
    return vecs


class StubEmbedder:
    """Minimal embedder that returns a fixed vector per known text, ignoring unknowns."""

    def __init__(self, chunk_map: dict[str, np.ndarray]) -> None:
        self._map = chunk_map
        self.dim = DIM

    def embed_one(self, text: str) -> np.ndarray:
        for key, vec in self._map.items():
            if key in text:
                return vec
        # Unknown → zero vector (will score low)
        return np.zeros(DIM, dtype=np.float32)


@pytest.fixture
def qdrant_client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def indexed_client(qdrant_client: QdrantClient) -> tuple[QdrantClient, dict[str, np.ndarray]]:
    embeddings = _stub_embeddings(FIXTURE_CHUNKS)
    chunk_map = {c["text"].split()[0]: embeddings[i] for i, c in enumerate(FIXTURE_CHUNKS)}

    indexer = QdrantIndexer(client=qdrant_client, collection=COLLECTION)
    indexer.create_collection(dim=DIM)
    indexer.index_chunks(FIXTURE_CHUNKS, embeddings)
    return qdrant_client, chunk_map


class TestQdrantIndexer:
    def test_collection_created(self, qdrant_client: QdrantClient) -> None:
        indexer = QdrantIndexer(client=qdrant_client, collection=COLLECTION)
        indexer.create_collection(dim=DIM)
        names = {c.name for c in qdrant_client.get_collections().collections}
        assert COLLECTION in names

    def test_chunk_count_matches(self, indexed_client: tuple) -> None:
        client, _ = indexed_client
        indexer = QdrantIndexer(client=client, collection=COLLECTION)
        assert indexer.collection_count() == len(FIXTURE_CHUNKS)

    def test_idempotent_create(self, qdrant_client: QdrantClient) -> None:
        indexer = QdrantIndexer(client=qdrant_client, collection=COLLECTION)
        indexer.create_collection(dim=DIM)
        indexer.create_collection(dim=DIM)  # second call should not raise
        assert indexer.collection_count() == 0  # no points yet

    def test_chunk_id_hash_stable(self) -> None:
        assert _chunk_id_to_uint64("aapl_c0") == _chunk_id_to_uint64("aapl_c0")

    def test_chunk_id_hash_distinct(self) -> None:
        assert _chunk_id_to_uint64("aapl_c0") != _chunk_id_to_uint64("jpm_c0")


class TestRetriever:
    def test_planted_query_returns_apple_chunk(self, indexed_client: tuple) -> None:
        """Known chunk must appear in top-k for a semantically matching query."""
        client, chunk_map = indexed_client
        embedder = StubEmbedder(chunk_map)
        retriever = Retriever(client=client, embedder=embedder, collection=COLLECTION, top_k=3)

        # Query vector matches "Apple" chunk (first word → index 0)
        query = "Apple net income"
        results = retriever.retrieve(query)

        assert len(results) > 0
        chunk_ids = [r["chunk_id"] for r in results]
        assert "aapl_c0" in chunk_ids, f"Apple chunk not found in top-k. Got: {chunk_ids}"

    def test_returns_at_most_top_k(self, indexed_client: tuple) -> None:
        client, chunk_map = indexed_client
        embedder = StubEmbedder(chunk_map)
        retriever = Retriever(client=client, embedder=embedder, collection=COLLECTION, top_k=2)
        results = retriever.retrieve("Apple income")
        assert len(results) <= 2

    def test_result_has_required_keys(self, indexed_client: tuple) -> None:
        client, chunk_map = indexed_client
        embedder = StubEmbedder(chunk_map)
        retriever = Retriever(client=client, embedder=embedder, collection=COLLECTION, top_k=1)
        results = retriever.retrieve("revenue")
        for r in results:
            assert "chunk_id" in r
            assert "text" in r
            assert "source_path" in r
            assert "score" in r

    def test_format_context_produces_string(self, indexed_client: tuple) -> None:
        client, chunk_map = indexed_client
        embedder = StubEmbedder(chunk_map)
        retriever = Retriever(client=client, embedder=embedder, collection=COLLECTION, top_k=2)
        chunks = retriever.retrieve("income")
        context = retriever.format_context(chunks)
        assert isinstance(context, str)
        assert len(context) > 0
        assert "[1]" in context

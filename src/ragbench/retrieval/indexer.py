"""Build and manage the Qdrant vector collection for the EDGAR corpus.

Indexing is deterministic and idempotent: re-running with the same chunks produces
the same collection. The chunk_id field is used as the Qdrant point ID (hashed to uint64).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ragbench"


def _chunk_id_to_uint64(chunk_id: str) -> int:
    """Stable hash of chunk_id string → uint64 for Qdrant point IDs."""
    return int(hashlib.sha256(chunk_id.encode()).hexdigest()[:16], 16)


class QdrantIndexer:
    def __init__(self, client: QdrantClient, collection: str = COLLECTION_NAME) -> None:
        self.client = client
        self.collection = collection

    def create_collection(self, dim: int, recreate: bool = False) -> None:
        """Create the Qdrant collection. Skips if it already exists (idempotent)."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            if recreate:
                self.client.delete_collection(self.collection)
                logger.info("Deleted existing collection '%s'", self.collection)
            else:
                logger.info("Collection '%s' already exists — skipping creation", self.collection)
                return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created collection '%s' (dim=%d)", self.collection, dim)

    def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        batch_size: int = 512,
    ) -> None:
        """Upload chunks with their embeddings to Qdrant.

        Args:
            chunks: List of chunk dicts; must have 'chunk_id' and 'text' keys.
            embeddings: Float32 array shape (N, dim), one row per chunk.
            batch_size: Points per upsert call.
        """
        assert len(chunks) == len(embeddings), "chunks and embeddings length mismatch"

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vecs = embeddings[start : start + batch_size]

            points = [
                PointStruct(
                    id=_chunk_id_to_uint64(c["chunk_id"]),
                    vector=vec.tolist(),
                    payload={
                        "chunk_id": c["chunk_id"],
                        "text": c["text"],
                        "source_path": c.get("source_path", ""),
                        "id": c.get("id", ""),
                    },
                )
                for c, vec in zip(batch_chunks, batch_vecs)
            ]
            self.client.upsert(collection_name=self.collection, points=points)

        logger.info("Indexed %d chunks into '%s'", len(chunks), self.collection)

    def collection_count(self) -> int:
        return self.client.get_collection(self.collection).points_count or 0

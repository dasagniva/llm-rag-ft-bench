"""Top-k retriever: embeds a query and fetches nearest chunks from Qdrant.

The retriever is plain Python over the Qdrant client — no LangChain, no framework.
Retrieved chunks include their source_path for citation in prompts.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient

from ragbench.retrieval.embedder import Embedder
from ragbench.retrieval.indexer import COLLECTION_NAME

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        collection: str = COLLECTION_NAME,
        top_k: int = 5,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict[str, str]]:
        """Return the top-k chunks most relevant to *query*.

        Returns:
            List of dicts with keys: chunk_id, text, source_path, score.
        """
        query_vec = self.embedder.embed_one(query)

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vec.tolist(),
            limit=self.top_k,
            with_payload=True,
        ).points

        return [
            {
                "chunk_id": str(r.payload.get("chunk_id", "")),
                "text": str(r.payload.get("text", "")),
                "source_path": str(r.payload.get("source_path", "")),
                "score": str(r.score),
            }
            for r in results
        ]

    def format_context(self, chunks: list[dict[str, str]]) -> str:
        """Format retrieved chunks into a context string for the prompt."""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[{i}] {c['text']}")
        return "\n\n".join(parts)

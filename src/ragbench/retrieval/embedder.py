"""Sentence-transformers embedding wrapper.

Model default: BAAI/bge-base-en-v1.5 (768-dim, strong on retrieval benchmarks).
Record final choice in DECISIONS.md at Phase 2 completion.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "auto") -> None:
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.dim: int = self.model.get_sentence_embedding_dimension()  # type: ignore[assignment]

    def embed(
        self, texts: list[str], batch_size: int = 256, show_progress: bool = False
    ) -> np.ndarray:
        """Embed a list of texts and return a float32 array of shape (N, dim)."""
        return self.model.encode(  # type: ignore[return-value]
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # L2-normalize for cosine via dot product
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text and return a 1-D float32 array."""
        return self.embed([text])[0]

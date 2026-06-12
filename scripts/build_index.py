#!/usr/bin/env python
"""Embed corpus chunks and upload to Qdrant.

Usage:
    # Start Qdrant first:
    docker compose up -d

    # Then build the index:
    uv run scripts/build_index.py --chunks data/raw/chunks.jsonl --config configs/rag.yaml

This script is idempotent: re-running with the same chunks updates/overwrites existing points.
Pass --recreate to drop and rebuild the collection from scratch.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
from qdrant_client import QdrantClient
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragbench.retrieval.embedder import Embedder
from ragbench.retrieval.indexer import QdrantIndexer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the Qdrant vector index")
    p.add_argument("--chunks", default="data/raw/chunks.jsonl")
    p.add_argument("--config", default="configs/rag.yaml")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--embed-batch-size", type=int, default=256)
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    retrieval_cfg = cfg.get("retrieval", {})
    embedding_model = retrieval_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
    qdrant_url = retrieval_cfg.get("qdrant_url", "http://localhost:6333")
    collection = retrieval_cfg.get("collection", "ragbench")

    logger.info("Loading chunks from %s …", args.chunks)
    with open(args.chunks) as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d chunks", len(chunks))

    logger.info("Loading embedding model: %s", embedding_model)
    embedder = Embedder(model_name=embedding_model)
    logger.info("Embedding model on device: %s  dim: %d", embedder.device, embedder.dim)

    client = QdrantClient(url=qdrant_url)
    indexer = QdrantIndexer(client=client, collection=collection)
    indexer.create_collection(dim=embedder.dim, recreate=args.recreate)

    # Stream embed→upload in upload_batch_size windows to avoid holding the full
    # embedding matrix (278K × 768 × 4B ≈ 856MB) in RAM simultaneously.
    upload_batch = args.batch_size  # chunks per Qdrant upsert call
    embed_batch = args.embed_batch_size  # texts per sentence-transformers encode call

    total = len(chunks)
    uploaded = 0
    with tqdm(total=total, desc="Embed + upload") as pbar:
        for upload_start in range(0, total, upload_batch):
            chunk_batch = chunks[upload_start : upload_start + upload_batch]
            texts = [c["text"] for c in chunk_batch]

            # Embed this upload window in smaller encode sub-batches
            import numpy as np

            sub_embeddings: list[np.ndarray] = []
            for e_start in range(0, len(texts), embed_batch):
                sub_embeddings.append(embedder.embed(texts[e_start : e_start + embed_batch]))
            embeddings = np.vstack(sub_embeddings)

            indexer.index_chunks(chunk_batch, embeddings, batch_size=upload_batch)
            uploaded += len(chunk_batch)
            pbar.update(len(chunk_batch))
            pbar.set_postfix(uploaded=uploaded)

    logger.info(
        "Index complete. Collection '%s' has %d points.", collection, indexer.collection_count()
    )


if __name__ == "__main__":
    main()

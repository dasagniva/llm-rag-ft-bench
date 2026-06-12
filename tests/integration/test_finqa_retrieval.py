"""Retrieval sanity test against the live FinQA/TAT-QA index (EXPERIMENT.md amendment
2026-06-12).

Tightened from "gold-doc-present" to "gold-span-present": for a sample of frozen
eval questions, the gold-supporting text/table-row span must appear verbatim
(whitespace-normalized) within the top-5 retrieved chunks — not merely the
correct source document.

Requires:
  - Qdrant running with the 'ragbench_finqa_tatqa' collection populated
    (docker compose up -d && uv run scripts/build_index.py --chunks
     data/raw/finqa_chunks.jsonl --config configs/rag.yaml)
  - data/raw/finqa_gold_spans.jsonl (uv run scripts/build_finqa_corpus.py)

Run with: .venv/bin/pytest tests/integration/ -m integration -v
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

qdrant_client_mod = pytest.importorskip(  # noqa: E402
    "qdrant_client",
    reason="qdrant_client import failed (likely grpcio/protobuf mismatch — see DECISIONS.md)",
)
QdrantClient = qdrant_client_mod.QdrantClient

from ragbench.corpus.finqa_tatqa import gold_span_present  # noqa: E402
from ragbench.retrieval.embedder import Embedder  # noqa: E402
from ragbench.retrieval.retriever import Retriever  # noqa: E402

ROOT = Path(__file__).parent.parent.parent
EVAL_SET = ROOT / "data" / "eval_manifest.jsonl"
GOLD_SPANS = ROOT / "data" / "raw" / "finqa_gold_spans.jsonl"
RAG_CONFIG = ROOT / "configs" / "rag.yaml"

SAMPLE_SIZE = 30
SAMPLE_SEED = 0
# Below the measured corpus-wide hit-rate@5 (~0.51); catches gross regressions
# (e.g. empty/misconfigured collection -> hit-rate near 0) without being brittle
# to per-sample noise on a 30-question subsample.
MIN_HIT_RATE_AT_5 = 0.30


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    if not GOLD_SPANS.exists():
        pytest.skip(f"{GOLD_SPANS} not found — run scripts/build_finqa_corpus.py")
    cfg = yaml.safe_load(RAG_CONFIG.read_text())["retrieval"]

    client = QdrantClient(url=cfg["qdrant_url"])
    collections = {c.name for c in client.get_collections().collections}
    if cfg["collection"] not in collections:
        pytest.skip(
            f"Qdrant collection '{cfg['collection']}' not found — run scripts/build_index.py"
        )

    embedder = Embedder(model_name=cfg["embedding_model"])
    return Retriever(client=client, embedder=embedder, collection=cfg["collection"], top_k=5)


@pytest.fixture(scope="module")
def sample_questions() -> list[dict]:
    rows = _load_jsonl(EVAL_SET)
    rng = random.Random(SAMPLE_SEED)
    return rng.sample(rows, min(SAMPLE_SIZE, len(rows)))


@pytest.fixture(scope="module")
def gold_spans() -> dict[str, list[str]]:
    return {r["id"]: r["gold_span_parts"] for r in _load_jsonl(GOLD_SPANS)}


class TestGoldSpanRetrieval:
    def test_gold_span_hit_rate_above_floor(
        self, retriever: Retriever, sample_questions: list[dict], gold_spans: dict[str, list[str]]
    ) -> None:
        hits = 0
        for row in sample_questions:
            parts = gold_spans.get(row["id"], [])
            retrieved = retriever.retrieve(row["question"])
            if gold_span_present(parts, [c["text"] for c in retrieved]):
                hits += 1

        hit_rate = hits / len(sample_questions)
        assert hit_rate >= MIN_HIT_RATE_AT_5, (
            f"gold-span hit-rate@5 = {hit_rate:.2f} on {len(sample_questions)} sampled "
            f"questions, below floor {MIN_HIT_RATE_AT_5} — retrieval may be misconfigured"
        )

    def test_retriever_returns_chunks_from_finqa_tatqa_corpus(self, retriever: Retriever) -> None:
        results = retriever.retrieve("What was the total sales in 2019?")
        assert len(results) > 0
        for r in results:
            assert r["chunk_id"].startswith(("finqa_", "tatqa_"))

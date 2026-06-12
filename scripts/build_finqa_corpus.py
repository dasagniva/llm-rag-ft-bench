#!/usr/bin/env python
"""Build the FinQA/TAT-QA-aligned retrieval corpus (EXPERIMENT.md amendment 2026-06-12).

Fetches the FinQA test split + TAT-QA dev split (the same splits the frozen
data/eval_manifest.jsonl was sampled from), builds one document per unique
source page/table, chunks them, and writes:

  - data/raw/finqa_chunks.jsonl     — chunks for scripts/build_index.py
  - data/raw/finqa_gold_spans.jsonl — per eval-question gold-supporting text spans,
                                       for scripts/retrieval_diagnostics.py

Usage:
    uv run scripts/build_finqa_corpus.py --config configs/finqa_corpus.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragbench.corpus.chunker import chunk_documents
from ragbench.corpus.finqa_tatqa import (
    finqa_documents,
    finqa_gold_span_parts,
    tatqa_documents,
    tatqa_gold_span_parts,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _fetch_json(url: str) -> list:
    logger.info("Fetching %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the FinQA/TAT-QA retrieval corpus")
    p.add_argument("--config", default="configs/finqa_corpus.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    finqa_raw = _fetch_json(cfg["finqa_test_url"])
    tatqa_raw = _fetch_json(cfg["tatqa_dev_url"])
    logger.info("FinQA test: %d QA examples", len(finqa_raw))
    logger.info("TAT-QA dev: %d table entries", len(tatqa_raw))

    fq_docs = finqa_documents(finqa_raw)
    tq_docs = tatqa_documents(tatqa_raw)
    logger.info("Unique FinQA source pages: %d", len(fq_docs))
    logger.info("Unique TAT-QA tables: %d", len(tq_docs))

    docs = fq_docs + tq_docs
    chunks = chunk_documents(docs, chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
    logger.info("Total documents: %d -> total chunks: %d", len(docs), len(chunks))

    out_chunks = Path(cfg["output_chunks_path"])
    out_chunks.parent.mkdir(parents=True, exist_ok=True)
    with open(out_chunks, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    logger.info("Wrote %s", out_chunks)

    # --- Gold spans, keyed by eval_manifest question id ---
    gold_spans: dict[str, list[str]] = {}
    for ex in finqa_raw:
        gold_spans[f"finqa_{ex['id']}"] = finqa_gold_span_parts(ex)
    for entry in tatqa_raw:
        for q in entry.get("questions", []):
            gold_spans[f"tatqa_{q['uid']}"] = tatqa_gold_span_parts(entry, q)

    out_spans = Path(cfg["output_gold_spans_path"])
    with open(out_spans, "w") as f:
        for qid, parts in gold_spans.items():
            f.write(json.dumps({"id": qid, "gold_span_parts": parts}) + "\n")
    logger.info("Wrote %s (%d questions)", out_spans, len(gold_spans))


if __name__ == "__main__":
    main()

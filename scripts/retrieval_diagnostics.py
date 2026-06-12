#!/usr/bin/env python
"""Retrieval diagnostic: gold-span hit-rate@k over the frozen eval set.

For each eval question, retrieves the top-k chunks from the FinQA/TAT-QA index
(configs/rag.yaml) and checks whether ALL gold-supporting text/table-row spans
(data/raw/finqa_gold_spans.jsonl, built by scripts/build_finqa_corpus.py) appear
verbatim (after whitespace normalization) within the retrieved chunk text.

This is the retrieval sanity check for EXPERIMENT.md amendment 2026-06-12:
"gold-span-present", not just "gold-doc-present" — the retrieved chunks must
actually contain the specific row/sentence the answer depends on.

Usage:
    uv run scripts/retrieval_diagnostics.py --config configs/rag.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow
import yaml
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragbench.corpus.finqa_tatqa import gold_span_present
from ragbench.retrieval.embedder import Embedder
from ragbench.retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

K_VALUES = (1, 3, 5)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieval gold-span hit-rate diagnostic")
    p.add_argument("--config", default="configs/rag.yaml")
    p.add_argument("--eval-set", default="data/eval_manifest.jsonl")
    p.add_argument("--gold-spans", default="data/raw/finqa_gold_spans.jsonl")
    p.add_argument("--output", default="reports/retrieval_diagnostics.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    retrieval_cfg = cfg.get("retrieval", {})

    eval_rows = load_jsonl(Path(args.eval_set))
    gold_spans = {r["id"]: r["gold_span_parts"] for r in load_jsonl(Path(args.gold_spans))}

    missing_gold = [r["id"] for r in eval_rows if r["id"] not in gold_spans]
    if missing_gold:
        raise ValueError(
            f"{len(missing_gold)} eval questions have no gold span entry, e.g. {missing_gold[:3]}"
        )

    embedder = Embedder(model_name=retrieval_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5"))
    client = QdrantClient(url=retrieval_cfg.get("qdrant_url", "http://localhost:6333"))
    max_k = max(K_VALUES)
    retriever = Retriever(
        client=client, embedder=embedder, collection=retrieval_cfg["collection"], top_k=max_k
    )

    hits = {k: 0 for k in K_VALUES}
    no_gold_span = 0
    per_question: list[dict] = []

    for row in eval_rows:
        parts = gold_spans[row["id"]]
        if not parts:
            no_gold_span += 1
        retrieved = retriever.retrieve(row["question"])
        retrieved_texts = [c["text"] for c in retrieved]
        row_hits = {}
        for k in K_VALUES:
            hit = gold_span_present(parts, retrieved_texts[:k])
            row_hits[f"hit@{k}"] = hit
            if hit:
                hits[k] += 1
        per_question.append({"id": row["id"], **row_hits})

    n = len(eval_rows)
    rates = {k: hits[k] / n for k in K_VALUES}

    print("\n" + "=" * 60)
    print("  Retrieval gold-span hit-rate diagnostic")
    print("=" * 60)
    print(f"  n = {n}, collection = {retrieval_cfg['collection']!r}")
    print(f"  questions with no gold span recorded: {no_gold_span}")
    for k in K_VALUES:
        print(f"  hit-rate@{k}: {rates[k]:.4f}  ({hits[k]}/{n})")
    print("=" * 60 + "\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Retrieval diagnostic: gold-span hit-rate",
        "",
        f"Collection: `{retrieval_cfg['collection']}` (n={n} eval questions)",
        "",
        "Gold-span-present = every gold-supporting text/table-row span for the "
        "question is found verbatim (whitespace-normalized) within the top-k "
        "retrieved chunks.",
        "",
        "| k | Hit rate | Hits / N |",
        "|---|---|---|",
    ]
    for k in K_VALUES:
        lines.append(f"| {k} | {rates[k]:.4f} | {hits[k]}/{n} |")
    lines.append("")
    lines.append(f"Questions with no recorded gold span: {no_gold_span}/{n}")
    lines.append("")
    out_path.write_text("\n".join(lines))
    logger.info("Wrote %s", out_path)

    mlflow.set_experiment(cfg.get("eval", {}).get("mlflow_experiment", "ragbench"))
    with mlflow.start_run(run_name="retrieval_diagnostics"):
        mlflow.log_param("collection", retrieval_cfg["collection"])
        mlflow.log_param("n_questions", n)
        for k in K_VALUES:
            mlflow.log_metric(f"gold_span_hit_rate_at_{k}", rates[k])
        mlflow.log_metric("no_gold_span_count", no_gold_span)
        table = {key: [row[key] for row in per_question] for key in per_question[0]}
        mlflow.log_table(data=table, artifact_file="retrieval_diagnostics.json")


if __name__ == "__main__":
    main()

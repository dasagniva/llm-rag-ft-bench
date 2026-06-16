#!/usr/bin/env python
"""Build the fine-tuning dataset from FinQA train and TAT-QA train splits.

Uses the FinQA *train* split and the TAT-QA *train* split — completely separate
from the frozen eval set (FinQA *test* + TAT-QA *dev*), so train/eval ID overlap
is zero by construction. The assertion at the end verifies this programmatically.

Usage:
    uv run scripts/build_finetune_set.py

Output:
    data/finetune_train.jsonl  — one JSON object per line with keys:
        id, question, reference_answer, source_dataset, answer_type

Fetches raw JSON directly from GitHub (same method as build_eval_set.py —
the HuggingFace datasets loader for FinQA/TAT-QA is broken on datasets>=3.x).
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FINQA_TRAIN_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json"
TATQA_TRAIN_URL = (
    "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/"
    "dataset_raw/tatqa_dataset_train.json"
)

OUT_PATH = Path("data/finetune_train.jsonl")
EVAL_PATH = Path("data/eval_manifest.jsonl")


def _fetch_json(url: str) -> list:
    logger.info("Fetching %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def _parse_finqa(raw: list) -> list[dict[str, str]]:
    items = []
    for ex in raw:
        try:
            q = ex.get("qa", {}).get("question", "").strip()
            a = str(ex.get("qa", {}).get("answer", "")).strip()
            if not q or not a:
                continue
            items.append(
                {
                    "id": f"finqa_{ex.get('id', '')}",
                    "question": q,
                    "reference_answer": a,
                    "source_dataset": "finqa",
                    "answer_type": "numeric" if any(c.isdigit() for c in a) else "text",
                }
            )
        except Exception:
            continue
    return items


def _parse_tatqa(raw: list) -> list[dict[str, str]]:
    items = []
    for entry in raw:
        for qa in entry.get("questions", []):
            try:
                q = qa.get("question", "").strip()
                answers = qa.get("answer", [])
                if isinstance(answers, str):
                    answers = [answers]
                a = " ".join(str(x) for x in answers).strip()
                if not q or not a:
                    continue
                scale = qa.get("scale", "").strip()
                if scale:
                    a = f"{a} {scale}"
                items.append(
                    {
                        "id": f"tatqa_{qa.get('uid', '')}",
                        "question": q,
                        "reference_answer": a,
                        "source_dataset": "tatqa",
                        "answer_type": qa.get("answer_type", "unknown"),
                    }
                )
            except Exception:
                continue
    return items


def _load_eval_ids() -> set[str]:
    if not EVAL_PATH.exists():
        logger.warning("Eval manifest not found at %s — skipping overlap check", EVAL_PATH)
        return set()
    ids: set[str] = set()
    with open(EVAL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def main() -> None:
    finqa_items = _parse_finqa(_fetch_json(FINQA_TRAIN_URL))
    logger.info("FinQA train: %d items", len(finqa_items))

    tatqa_items = _parse_tatqa(_fetch_json(TATQA_TRAIN_URL))
    logger.info("TAT-QA train: %d items", len(tatqa_items))

    combined = finqa_items + tatqa_items
    logger.info("Total fine-tuning examples: %d", len(combined))

    # Zero-overlap assertion (EXPERIMENT.md requirement; expected to always pass
    # because train and eval use completely different dataset splits).
    eval_ids = _load_eval_ids()
    if eval_ids:
        train_ids = {ex["id"] for ex in combined}
        overlap = train_ids & eval_ids
        if overlap:
            logger.error(
                "TRAIN/EVAL OVERLAP DETECTED — %d IDs appear in both sets: %s …",
                len(overlap),
                list(overlap)[:5],
            )
            sys.exit(1)
        logger.info(
            "Zero-overlap check passed: no train IDs found in eval set (n_eval=%d)", len(eval_ids)
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for item in combined:
            f.write(json.dumps(item) + "\n")

    logger.info("Fine-tuning set written: %d examples → %s", len(combined), OUT_PATH)
    logger.info("NEXT: uv run scripts/finetune.py --config configs/finetune.yaml")


if __name__ == "__main__":
    main()

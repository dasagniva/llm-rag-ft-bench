#!/usr/bin/env python
"""Sample and freeze the evaluation set from FinQA and TAT-QA.

Usage:
    uv run scripts/build_eval_set.py --config configs/eval.yaml

Output:
    data/eval_manifest.jsonl  — committed; one JSON object per line with keys:
        id, question, reference_answer, source_dataset, answer_type

This script is run ONCE. The resulting file is committed and never regenerated
(changing the eval set after seeing results is experimental malpractice).

Source data fetched directly from GitHub (raw JSON) because the HuggingFace
datasets versions of FinQA and TAT-QA still use deprecated loading scripts
that datasets>=3.x no longer supports.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FINQA_TEST_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json"
TATQA_DEV_URL = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_dev.json"
# Note: tatqa_dataset_test.json withholds answers (competition format); dev set is used instead.


def _fetch_json(url: str) -> list:
    logger.info("Fetching %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the frozen evaluation set")
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing eval_manifest.jsonl (use only before any results have been seen)",
    )
    return p.parse_args()


def _parse_finqa(raw: list) -> list[dict[str, str]]:
    """Extract QA pairs from FinQA raw JSON.

    FinQA format: list of dicts with keys id, pre_text, post_text, table,
    qa {question, answer, steps, program}.
    """
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
    """Extract QA pairs from TAT-QA raw JSON (dev set).

    TAT-QA format: list of table+paragraph entries, each with a 'questions' list.
    Each question dict has uid, question, answer (list), answer_type, scale.
    Scale (million, thousand, billion, percent) is appended to numeric answers.
    """
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
                # Append scale so reference matches expected model output
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


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_path = Path(cfg.get("eval_set_path", "data/eval_manifest.jsonl"))

    if out_path.exists() and not args.force:
        logger.error(
            "%s already exists. The eval set is frozen — use --force only if you have NOT "
            "yet seen any eval results.",
            out_path,
        )
        sys.exit(1)

    seed = cfg.get("sample_seed", 0)
    rng = random.Random(seed)

    # --- FinQA ---
    finqa_n = cfg.get("finqa_sample_n", 180)
    finqa_items = _parse_finqa(_fetch_json(FINQA_TEST_URL))
    rng.shuffle(finqa_items)
    finqa_sample = finqa_items[:finqa_n]
    logger.info("FinQA: %d usable → sampled %d", len(finqa_items), len(finqa_sample))

    # --- TAT-QA ---
    tatqa_n = cfg.get("tatqa_sample_n", 120)
    tatqa_items = _parse_tatqa(_fetch_json(TATQA_DEV_URL))
    rng.shuffle(tatqa_items)
    tatqa_sample = tatqa_items[:tatqa_n]
    logger.info("TAT-QA: %d usable → sampled %d", len(tatqa_items), len(tatqa_sample))

    combined = finqa_sample + tatqa_sample
    rng.shuffle(combined)

    if len(combined) < cfg.get("target_n", 300):
        logger.warning(
            "Eval set size %d is below target %d. Consider adjusting sample_n values in %s.",
            len(combined),
            cfg.get("target_n", 300),
            args.config,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for item in combined:
            f.write(json.dumps(item) + "\n")

    logger.info("Eval set written: %d questions → %s", len(combined), out_path)
    logger.info("Seed: %d  FinQA: %d  TAT-QA: %d", seed, len(finqa_sample), len(tatqa_sample))
    logger.info(
        "NEXT STEP [HUMAN]: manually audit ~20 sampled items and record notes in EXPERIMENT.md §9."
    )


if __name__ == "__main__":
    main()

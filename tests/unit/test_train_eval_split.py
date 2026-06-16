"""Assert zero ID overlap between the fine-tuning set and the frozen eval set.

The overlap is zero by construction (train/test and train/dev splits are disjoint)
but this test verifies it programmatically, guarding against any future accidental
cross-contamination.

Skips if data/finetune_train.jsonl is not yet built (run build_finetune_set.py first).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EVAL_MANIFEST = Path("data/eval_manifest.jsonl")
FINETUNE_TRAIN = Path("data/finetune_train.jsonl")


def _load_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


class TestOverlapDetectionLogic:
    """Pure unit tests for the overlap detection logic — no file I/O."""

    def test_disjoint_sets_have_no_overlap(self) -> None:
        a = {"finqa_1", "finqa_2", "tatqa_3"}
        b = {"finqa_10", "tatqa_20"}
        assert a & b == set()

    def test_overlapping_sets_are_detected(self) -> None:
        a = {"finqa_1", "finqa_2"}
        b = {"finqa_2", "tatqa_3"}
        assert a & b == {"finqa_2"}

    def test_empty_sets_have_no_overlap(self) -> None:
        assert set() & {"finqa_1"} == set()


@pytest.mark.skipif(
    not FINETUNE_TRAIN.exists(),
    reason="data/finetune_train.jsonl not built yet — run scripts/build_finetune_set.py",
)
def test_no_train_eval_id_overlap() -> None:
    """No ID in the fine-tuning set may appear in the frozen eval set."""
    assert EVAL_MANIFEST.exists(), f"Eval manifest missing: {EVAL_MANIFEST}"

    eval_ids = _load_ids(EVAL_MANIFEST)
    train_ids = _load_ids(FINETUNE_TRAIN)

    assert len(eval_ids) > 0, "Eval manifest is empty"
    assert len(train_ids) > 0, "Fine-tuning set is empty"

    overlap = train_ids & eval_ids
    assert overlap == set(), (
        f"Train/eval ID overlap detected — {len(overlap)} IDs in both sets: {sorted(overlap)[:10]}"
    )


@pytest.mark.skipif(
    not FINETUNE_TRAIN.exists(),
    reason="data/finetune_train.jsonl not built yet — run scripts/build_finetune_set.py",
)
def test_finetune_set_size_reasonable() -> None:
    """Fine-tuning set should contain at least 1,000 examples (sanity guard)."""
    train_ids = _load_ids(FINETUNE_TRAIN)
    assert len(train_ids) >= 1_000, (
        f"Fine-tuning set has only {len(train_ids)} examples — expected ≥ 1,000"
    )

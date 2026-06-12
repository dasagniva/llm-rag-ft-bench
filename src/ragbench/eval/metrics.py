"""Per-question evaluation metrics.

All functions are pure (no I/O, no state). Inputs are strings; outputs are floats in [0, 1].
These functions are called per-question by the eval runner and also used in stats.py.

Normalization follows the SQuAD/FinQA convention:
  lowercase → strip articles → strip punctuation → collapse whitespace → normalize numbers.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import TYPE_CHECKING

from ragbench.eval.normalize import numeric_exact_match

if TYPE_CHECKING:
    import numpy as np


def normalize_answer(text: str) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace, normalize number formatting."""
    text = text.lower()

    # Strip articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Remove punctuation except for decimal points inside numbers (keep "3.14", drop "...")
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)  # dots not surrounded by digits → space
    text = text.translate(str.maketrans("", "", string.punctuation.replace(".", "")))

    # Normalize number formatting: "1,234" → "1234"; "10%" → "10 %"
    text = re.sub(r"(\d),(\d)", r"\1\2", text)

    return " ".join(text.split())


def exact_match(prediction: str, reference: str, numeric_rel_tol: float = 1e-3) -> float:
    """Return 1.0 if prediction matches reference, else 0.0.

    If both *prediction* and *reference* contain a parseable number (after stripping
    currency symbols, commas, scale words, and spelled-out numbers — see
    `eval.normalize`), they are compared with relative tolerance *numeric_rel_tol*.
    Otherwise falls back to string equality after `normalize_answer`.
    """
    numeric = numeric_exact_match(prediction, reference, rel_tol=numeric_rel_tol)
    if numeric is not None:
        return numeric
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between normalized prediction and reference.

    Returns 0.0 when either side is empty after normalization.
    This is the standard SQuAD token F1 metric.
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common: Counter[str] = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def faithfulness(answer: str, context: str) -> float:
    """Lexical faithfulness proxy: fraction of answer bigrams present in the context.

    This is a local, zero-cost proxy for LLM-based faithfulness scoring. For single-token
    answers the metric degrades to unigram presence. See DECISIONS.md for the rationale
    for using a lexical proxy instead of a judge model.

    Range: [0, 1]. 1.0 means every answer bigram is supported by the context text.
    Only meaningful for RAG and ft+rag configurations where context is non-empty.
    """
    if not context:
        return 0.0

    a_tokens = normalize_answer(answer).split()
    c_tokens = normalize_answer(context).split()

    if not a_tokens:
        return 0.0

    if len(a_tokens) == 1:
        return float(a_tokens[0] in set(c_tokens))

    c_bigram_set = set(zip(c_tokens, c_tokens[1:]))
    a_bigrams = list(zip(a_tokens, a_tokens[1:]))
    matches = sum(1 for bg in a_bigrams if bg in c_bigram_set)
    return matches / len(a_bigrams)


def answer_relevance(
    answer: str,
    question: str,
    answer_embedding: np.ndarray,
    question_embedding: np.ndarray,
) -> float:
    """Cosine similarity between pre-computed answer and question embeddings.

    Embeddings must be L2-normalised (Embedder always normalises).
    Returns a value in [-1, 1]; practically in [0, 1] for semantic similarity.
    """
    import numpy as np

    a = np.asarray(answer_embedding, dtype=np.float32)
    q = np.asarray(question_embedding, dtype=np.float32)
    a_norm = np.linalg.norm(a)
    q_norm = np.linalg.norm(q)
    if a_norm == 0 or q_norm == 0:
        return 0.0
    return float(np.dot(a / a_norm, q / q_norm))


def compute_metrics(
    prediction: str, reference: str, numeric_rel_tol: float = 1e-3
) -> dict[str, float]:
    """Compute base per-question metrics (EM + F1). Used for all configurations."""
    return {
        "exact_match": exact_match(prediction, reference, numeric_rel_tol=numeric_rel_tol),
        "token_f1": token_f1(prediction, reference),
    }


def compute_rag_metrics(
    prediction: str,
    reference: str,
    context: str,
    answer_embedding: np.ndarray | None = None,
    question_embedding: np.ndarray | None = None,
    numeric_rel_tol: float = 1e-3,
) -> dict[str, float]:
    """Compute all metrics for RAG configurations (adds faithfulness + answer relevance)."""
    metrics = compute_metrics(prediction, reference, numeric_rel_tol=numeric_rel_tol)
    metrics["faithfulness"] = faithfulness(prediction, context)
    if answer_embedding is not None and question_embedding is not None:
        metrics["answer_relevance"] = answer_relevance(
            prediction, reference, answer_embedding, question_embedding
        )
    else:
        metrics["answer_relevance"] = float("nan")
    return metrics

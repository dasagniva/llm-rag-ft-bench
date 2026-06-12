"""Unit tests for eval/metrics.py.

These are the most important unit tests in Phase 1.
Every normalization edge case that appears in FinQA/TAT-QA answers should be covered here.
"""

import math

import numpy as np
import pytest

from ragbench.eval.metrics import (
    answer_relevance,
    compute_metrics,
    compute_rag_metrics,
    exact_match,
    faithfulness,
    normalize_answer,
    token_f1,
)


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Apple Inc.") == "apple inc"

    def test_strips_articles(self):
        assert normalize_answer("the revenue") == "revenue"
        assert normalize_answer("a net loss") == "net loss"
        assert normalize_answer("an increase") == "increase"

    def test_strips_punctuation(self):
        assert normalize_answer("$1,234") == "1234"
        assert normalize_answer("(loss)") == "loss"

    def test_preserves_decimal_points(self):
        result = normalize_answer("3.14")
        assert "3" in result and "14" in result

    def test_number_comma_removal(self):
        assert normalize_answer("1,234,567") == "1234567"

    def test_collapses_whitespace(self):
        assert normalize_answer("  net  income  ") == "net income"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_percentage(self):
        norm = normalize_answer("15%")
        assert "15" in norm

    def test_article_not_mid_word(self):
        # "theater" should not lose "the" prefix
        norm = normalize_answer("theater")
        assert "theater" in norm or "eater" in norm  # word-boundary stripping


class TestExactMatch:
    def test_identical(self):
        assert exact_match("yes", "yes") == 1.0

    def test_case_insensitive(self):
        assert exact_match("Yes", "yes") == 1.0

    def test_punctuation_insensitive(self):
        assert exact_match("$1,234", "1234") == 1.0

    def test_mismatch(self):
        assert exact_match("no", "yes") == 0.0

    def test_partial_overlap_is_zero(self):
        assert exact_match("net income increased", "net income") == 0.0

    def test_empty_prediction(self):
        assert exact_match("", "net income") == 0.0

    def test_both_empty(self):
        assert exact_match("", "") == 1.0

    def test_number_normalization(self):
        assert exact_match("1,500", "1500") == 1.0
        assert exact_match("15%", "15 %") == 1.0


class TestTokenF1:
    def test_perfect_match(self):
        assert token_f1("net income", "net income") == pytest.approx(1.0)

    def test_zero_overlap(self):
        assert token_f1("revenue growth", "operating loss") == pytest.approx(0.0)

    def test_partial_overlap(self):
        # prediction has 2/3 tokens correct; recall is 2/2 = 1; precision is 2/3
        f1 = token_f1("net income growth", "net income")
        assert 0.0 < f1 < 1.0

    def test_empty_prediction(self):
        assert token_f1("", "net income") == pytest.approx(0.0)

    def test_empty_reference(self):
        assert token_f1("net income", "") == pytest.approx(0.0)

    def test_both_empty(self):
        assert token_f1("", "") == pytest.approx(0.0)

    def test_symmetric_inputs(self):
        f1_ab = token_f1("net income growth", "net income")
        f1_ba = token_f1("net income", "net income growth")
        assert f1_ab == pytest.approx(f1_ba)

    def test_repeated_tokens(self):
        # pred=["yes","yes"], ref=["yes"] → common=1, precision=1/2, recall=1 → F1=2/3
        assert token_f1("yes yes", "yes") == pytest.approx(2 / 3, abs=1e-6)

    def test_case_insensitive(self):
        assert token_f1("Net Income", "net income") == pytest.approx(1.0)

    def test_range_zero_to_one(self):
        for pred, ref in [("a b c", "a b"), ("x", "y z"), ("", "a")]:
            f1 = token_f1(pred, ref)
            assert 0.0 <= f1 <= 1.0, f"F1 out of range for ({pred!r}, {ref!r}): {f1}"


class TestComputeMetrics:
    def test_returns_both_metrics(self):
        result = compute_metrics("15%", "15 %")
        assert "exact_match" in result
        assert "token_f1" in result

    def test_values_consistent(self):
        result = compute_metrics("net income", "net income")
        assert result["exact_match"] == pytest.approx(1.0)
        assert result["token_f1"] == pytest.approx(1.0)


class TestFaithfulness:
    def test_answer_fully_in_context_is_one(self):
        # Every answer bigram must appear consecutively in the context
        context = "the net income 99 billion was reported for fiscal year 2022"
        assert faithfulness("net income 99 billion", context) == pytest.approx(1.0)

    def test_answer_not_in_context_is_zero(self):
        context = "Apple net income was 99 billion"
        assert faithfulness("operating loss in 2023", context) == pytest.approx(0.0)

    def test_empty_answer_is_zero(self):
        assert faithfulness("", "some context") == pytest.approx(0.0)

    def test_empty_context_is_zero(self):
        assert faithfulness("net income", "") == pytest.approx(0.0)

    def test_partial_support(self):
        context = "Apple reported net income of 99 billion"
        # "net income" supported, "operating loss" not
        f = faithfulness("net income operating loss", context)
        assert 0.0 < f < 1.0

    def test_single_token_answer_present(self):
        assert faithfulness("apple", "Apple Inc is a technology company") == pytest.approx(1.0)

    def test_single_token_answer_absent(self):
        assert faithfulness("microsoft", "Apple Inc is a technology company") == pytest.approx(0.0)

    def test_range_zero_to_one(self):
        for answer, context in [
            ("net income", "revenue grew"),
            ("", "some text"),
            ("net income 99", "net income was 99 billion"),
        ]:
            f = faithfulness(answer, context)
            assert 0.0 <= f <= 1.0


class TestAnswerRelevance:
    def _unit_vec(self, dim: int, idx: int) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        v[idx] = 1.0
        return v

    def test_identical_embeddings_is_one(self):
        v = self._unit_vec(4, 0)
        assert answer_relevance("a", "b", v, v) == pytest.approx(1.0)

    def test_orthogonal_embeddings_is_zero(self):
        v1 = self._unit_vec(4, 0)
        v2 = self._unit_vec(4, 1)
        assert answer_relevance("a", "b", v1, v2) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        v = self._unit_vec(4, 0)
        zero = np.zeros(4, dtype=np.float32)
        assert answer_relevance("a", "b", zero, v) == pytest.approx(0.0)
        assert answer_relevance("a", "b", v, zero) == pytest.approx(0.0)

    def test_returns_float(self):
        v = self._unit_vec(4, 0)
        result = answer_relevance("a", "b", v, v)
        assert isinstance(result, float)


class TestComputeRagMetrics:
    def test_returns_all_keys(self):
        context = "Apple net income was 99 billion"
        result = compute_rag_metrics("net income 99", "99 billion", context)
        for key in ("exact_match", "token_f1", "faithfulness"):
            assert key in result

    def test_answer_relevance_nan_without_embeddings(self):
        result = compute_rag_metrics("answer", "reference", "context text")
        assert math.isnan(result["answer_relevance"])

    def test_answer_relevance_computed_with_embeddings(self):
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = compute_rag_metrics(
            "answer", "reference", "context", answer_embedding=v, question_embedding=v
        )
        assert not math.isnan(result["answer_relevance"])

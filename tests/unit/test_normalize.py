"""Unit tests for eval/normalize.py — numeric extraction and tolerance-based EM.

These cover the edge cases observed in actual base-run predictions: currency
symbols, scale words, spelled-out numbers, parenthesized negatives, percent
signs, and the relative-tolerance boundary.
"""

import pytest

from ragbench.eval.metrics import exact_match
from ragbench.eval.normalize import (
    extract_number,
    is_numeric_string,
    numeric_exact_match,
    words_to_digits,
)


class TestWordsToDigits:
    def test_single_word(self):
        assert words_to_digits("two") == "2"

    def test_in_sentence(self):
        assert words_to_digits("there are two types") == "there are 2 types"

    def test_compound_number(self):
        assert words_to_digits("twenty three") == "23"

    def test_with_hundred(self):
        assert words_to_digits("two hundred fifty three") == "253"

    def test_with_scale_word(self):
        assert words_to_digits("four million two hundred thousand") == "4200000"

    def test_standalone_scale_word_untouched(self):
        # "billion" with no preceding units/tens word is ordinary prose
        assert words_to_digits("a billion dollar deal") == "a billion dollar deal"

    def test_non_number_words_untouched(self):
        assert words_to_digits("net income increased") == "net income increased"


class TestExtractNumber:
    def test_plain_integer(self):
        assert extract_number("1167") == 1167.0

    def test_currency_symbol(self):
        assert extract_number("$4.2") == 4.2

    def test_comma_thousands(self):
        assert extract_number("1,234,567") == 1234567.0

    def test_parenthesized_negative(self):
        assert extract_number("(31.47)") == -31.47

    def test_leading_minus(self):
        assert extract_number("-31.47") == -31.47

    def test_percent_sign_kept_as_points(self):
        assert extract_number("214.36%") == 214.36

    def test_scale_word_million(self):
        assert extract_number("4.2 million") == 4_200_000.0

    def test_scale_word_billion(self):
        assert extract_number("1.5 billion") == 1_500_000_000.0

    def test_spelled_out_number(self):
        assert extract_number("There are **two** types of assets.") == 2.0

    def test_last_number_in_free_text(self):
        text = "In 2019 the value was 100, but by 2020 it had grown to 445.0"
        assert extract_number(text) == 445.0

    def test_no_number_returns_none(self):
        assert extract_number("Please provide the necessary data.") is None

    def test_empty_string_returns_none(self):
        assert extract_number("") is None

    def test_currency_pound_symbol(self):
        assert extract_number("£445.0") == 445.0


class TestIsNumericString:
    def test_bare_integer(self):
        assert is_numeric_string("1167")

    def test_negative_with_trailing_comma(self):
        assert is_numeric_string("-31.47,")

    def test_percent(self):
        assert is_numeric_string("214.36%")

    def test_pence_suffix(self):
        assert is_numeric_string("445.0p")

    def test_multiple_numbers_is_not_numeric(self):
        assert not is_numeric_string("2019 2018")

    def test_long_free_text_is_not_numeric(self):
        text = (
            "Interest expense increased in fiscal 2019 due to higher average "
            "borrowings of $10.0 billion of senior notes."
        )
        assert not is_numeric_string(text)

    def test_empty_string(self):
        assert not is_numeric_string("")


class TestNumericExactMatch:
    def test_year_like_numbers_require_exact_equality(self):
        # 2021 vs 2019: relative diff ~0.099% is within default 1e-3 tolerance,
        # but these are different years and must NOT match.
        assert numeric_exact_match("2021", "2019") == 0.0
        assert numeric_exact_match("2019", "2019") == 1.0

    def test_free_text_gold_does_not_spuriously_match(self):
        # Both sides end in a financial figure, but the gold is free text, not a
        # numeric answer -> must fall back to string match (None here), not 1.0.
        pred = (
            "Interest expense increased due to higher borrowings of $10.0 billion "
            "of senior notes during fiscal 2019."
        )
        gold = (
            "Interest expense increased due to the maturities and repayments of "
            "$6.0 billion of senior notes during fiscal 2018."
        )
        assert numeric_exact_match(pred, gold) is None

    def test_exact_equal(self):
        assert numeric_exact_match("1167", "1167") == 1.0

    def test_within_tolerance(self):
        # 1167.5 vs 1167 -> rel diff ~4.3e-4, within default 1e-3
        assert numeric_exact_match("1167.5", "1167") == 1.0

    def test_outside_tolerance(self):
        # 215.4 vs 214.36 -> rel diff ~4.85e-3, outside default 1e-3
        assert numeric_exact_match("215.4%", "214.36%") == 0.0

    def test_tolerance_boundary(self):
        # exactly at rel_tol -> should match (<=)
        ref = 1000.0
        pred = ref * (1 + 1e-3)
        assert numeric_exact_match(str(pred), str(ref), rel_tol=1e-3) == 1.0

    def test_zero_reference_exact(self):
        assert numeric_exact_match("0", "0") == 1.0

    def test_zero_reference_nonzero_prediction(self):
        assert numeric_exact_match("1", "0") == 0.0

    def test_non_numeric_returns_none(self):
        assert numeric_exact_match("yes", "no") is None

    def test_one_sided_non_numeric_returns_none(self):
        assert numeric_exact_match("Please provide more data.", "1167") is None

    def test_word_number_matches_digit(self):
        assert numeric_exact_match("There are **two** types.", "2") == 1.0

    def test_configurable_tolerance(self):
        # 1100 vs 1000 -> rel diff 0.1; matches with rel_tol=0.2 but not 0.01
        assert numeric_exact_match("1100", "1000", rel_tol=0.2) == 1.0
        assert numeric_exact_match("1100", "1000", rel_tol=0.01) == 0.0


class TestExactMatchIntegration:
    def test_unit_mismatch_no_false_positive(self):
        # Different magnitudes must not accidentally match
        assert exact_match("2.1%", "11%") == 0.0

    def test_refusal_stays_zero(self):
        assert exact_match("Please provide the necessary data or context.", "1167") == 0.0

    def test_currency_and_scale(self):
        assert exact_match("$4.2 million", "4200000") == 1.0

    def test_spelled_number_vs_digit(self):
        assert exact_match("There are **two** types of finite-lived intangible assets.", "2") == 1.0

    def test_existing_string_fallback_unchanged(self):
        assert exact_match("yes", "yes") == 1.0
        assert exact_match("no", "yes") == 0.0
        assert exact_match("net income increased", "net income") == 0.0

    @pytest.mark.parametrize("rel_tol", [1e-3, 1e-2, 0.05])
    def test_tolerance_is_configurable_end_to_end(self, rel_tol):
        result = exact_match("1100", "1000", numeric_rel_tol=rel_tol)
        assert result == float(rel_tol >= 0.1)

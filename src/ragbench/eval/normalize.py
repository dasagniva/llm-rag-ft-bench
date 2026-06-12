"""Numeric answer extraction and tolerance-based exact match for FinQA / TAT-QA.

FinQA/TAT-QA gold answers are short numeric strings (e.g. "1167", "-31.47", "214.36%",
"445.0p"), while model generations are free-text, often with currency symbols, comma
thousands separators, spelled-out numbers ("two"), and scale words ("4.2 million").

This module is pure (no I/O) and is used by `eval.metrics.exact_match` to apply a
relative-tolerance numeric comparison when both sides parse as numbers, falling back
to the existing string-normalization exact match otherwise.
"""

from __future__ import annotations

import re

_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}
_NUMBER_VOCAB = set(_UNITS) | set(_TENS) | set(_SCALES) | {"and"}

# A number-with-optional-scale, e.g. "1,234", "(31.47)", "-3.4%", "215.4"
_NUM_RE = re.compile(r"\(?-?\d+(?:\.\d+)?\)?%?")
_SCALE_SUFFIX_RE = re.compile(r"\s*(thousand|million|billion|trillion|percent|pct)\b")

# Matches a string that, after normalization, IS a number (optionally with a trailing
# unit suffix) and nothing else — e.g. "1167", "-31.47", "214.36%", "445.0p".
_NUMERIC_ONLY_RE = re.compile(
    r"^\(?-?\d+(?:\.\d+)?\)?\s*(%|percent|pct|thousand|million|billion|trillion|p)?$",
    re.IGNORECASE,
)

_YEAR_MIN, _YEAR_MAX = 1900, 2100


def _eval_number_words(run: list[str]) -> float | None:
    """Evaluate a run of number-word tokens (e.g. ["two", "hundred"]) to a float.

    Returns None unless at least one units/tens word is present — this prevents
    standalone scale words ("billion" in "billion dollar deal") from being
    misread as numbers.
    """
    total = 0.0
    current = 0.0
    seen_unit = False
    for word in run:
        if word == "and":
            continue
        if word in _UNITS:
            current += _UNITS[word]
            seen_unit = True
        elif word in _TENS:
            current += _TENS[word]
            seen_unit = True
        elif word == "hundred":
            current = (current or 1) * 100
        elif word in _SCALES:
            total += (current or 1) * _SCALES[word]
            current = 0
    if not seen_unit:
        return None
    return total + current


def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return repr(value)


def words_to_digits(text: str) -> str:
    """Replace runs of spelled-out numbers (e.g. "twenty three") with digit strings.

    Standalone scale words ("billion" with no preceding units/tens word) are left
    untouched, since they are usually part of ordinary prose ("a billion-dollar deal").
    """
    tokens = re.findall(r"[a-z']+|[^a-z\s]+|\s+", text)
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in _NUMBER_VOCAB:
            j = i
            run: list[str] = []
            while j < n:
                t = tokens[j]
                if t in _NUMBER_VOCAB:
                    run.append(t)
                    j += 1
                elif t.isspace() and j + 1 < n and tokens[j + 1] in _NUMBER_VOCAB:
                    j += 1  # skip whitespace between number words
                else:
                    break
            value = _eval_number_words(run)
            if value is not None:
                out.append(_format_number(value))
                i = j
                continue
        out.append(tok)
        i += 1
    return "".join(out)


def extract_number(text: str) -> float | None:
    """Parse the final numeric value mentioned in *text*, or None if there is none.

    Handles:
      - currency symbols ($, £, €, ¥) and comma thousands separators
      - parenthesized negatives, e.g. "(31.47)" -> -31.47
      - percent signs, e.g. "214.36%" -> 214.36 (kept as percentage points)
      - scale words, e.g. "4.2 million" -> 4_200_000
      - spelled-out numbers, e.g. "two hundred" -> 200

    For free-text generations containing multiple numbers, the LAST number found
    is taken as "the final answer" (the common pattern in chain-of-thought output).
    """
    if not text:
        return None
    t = text.lower().replace("**", "").replace("*", "")
    t = words_to_digits(t)
    t = re.sub(r"[$£€¥]", "", t)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)

    results: list[float] = []
    for m in _NUM_RE.finditer(t):
        raw = m.group(0)
        is_pct = raw.endswith("%")
        core = raw.strip("()%")
        is_negative = raw.startswith("(") and raw.endswith(")")
        try:
            val = float(core)
        except ValueError:
            continue
        if is_negative:
            val = -val
        if not is_pct:
            rest = t[m.end() :]
            scale_m = _SCALE_SUFFIX_RE.match(rest)
            if scale_m:
                word = scale_m.group(1)
                if word in _SCALES:
                    val *= _SCALES[word]
        results.append(val)

    return results[-1] if results else None


def is_numeric_string(text: str) -> bool:
    """True if *text*, once normalized, is a bare number (with optional unit suffix).

    Used to gate numeric tolerance matching to gold answers that ARE a number
    (e.g. "1167", "214.36%", "445.0p") as opposed to free-text gold answers that
    merely happen to contain a number ("...issuance of $10.0 billion of senior
    notes..."). Without this gate, two long free-text answers could spuriously
    match if `extract_number` happens to pick the same trailing number from each.
    """
    if not text:
        return False
    t = text.strip().lower().replace("**", "").replace("*", "")
    t = words_to_digits(t)
    t = re.sub(r"[$£€¥]", "", t).strip()
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    t = t.strip().rstrip(",.")
    return bool(_NUMERIC_ONLY_RE.match(t))


def _looks_like_year(value: float) -> bool:
    return value.is_integer() and _YEAR_MIN <= value <= _YEAR_MAX


def numeric_exact_match(prediction: str, reference: str, rel_tol: float = 1e-3) -> float | None:
    """Tolerance-based exact match for numeric answers.

    Returns 1.0/0.0 if *reference* is a bare number (see `is_numeric_string`) and
    *prediction* contains a parseable number, comparing with relative tolerance
    *rel_tol*. Returns None otherwise, so callers can fall back to string-based
    exact match.

    Numbers that look like calendar years (integers in [1900, 2100]) are compared
    for exact equality regardless of *rel_tol* — a 0.1% relative tolerance on a
    4-digit year is large enough to treat adjacent years (e.g. 2019 vs 2020) as
    "matching", which is never the intended behaviour for date-valued answers.
    """
    if not is_numeric_string(reference):
        return None
    pred_num = extract_number(prediction)
    ref_num = extract_number(reference)
    if pred_num is None or ref_num is None:
        return None
    if _looks_like_year(pred_num) and _looks_like_year(ref_num):
        return float(pred_num == ref_num)
    if ref_num == 0:
        return float(abs(pred_num) <= rel_tol)
    return float(abs(pred_num - ref_num) <= rel_tol * abs(ref_num))

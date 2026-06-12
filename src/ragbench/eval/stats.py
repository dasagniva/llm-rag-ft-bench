"""
Statistical comparison of paired LLM evaluation results.

All public functions are pure: numpy array inputs → dataclass outputs.
No I/O, no side effects, no mutable state. Seeds guarantee bit-for-bit
reproducibility across runs and machines.

Pre-registered protocol (EXPERIMENT.md):
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Paired bootstrap   B=10,000  CIs on per-system metrics and Δ        │
  │ McNemar exact                Paired binary outcomes (exact match)   │
  │ Paired permutation B=10,000  Continuous metrics (F1, faithfulness)  │
  │ Holm–Bonferroni              FWER control across contrasts/metric   │
  └──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sub_seed(base: int, *parts: str) -> int:
    """Derive a deterministic integer sub-seed from a base seed and string tags.

    Uses MD5 (8 hex chars → 32-bit integer) XOR'd with *base* so that every
    (base, tags) combination maps to a unique, reproducible seed without
    relying on Python's PYTHONHASHSEED-sensitive built-in hash().
    """
    digest = hashlib.md5(":".join(parts).encode()).hexdigest()[:8]
    return base ^ int(digest, 16)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap 95% CI for the mean of one metric on one configuration."""

    metric: str
    config: str
    n: int  # number of questions
    point_estimate: float
    ci_lower: float  # 2.5th percentile of bootstrap distribution
    ci_upper: float  # 97.5th percentile
    b: int  # resamples used

    def __str__(self) -> str:
        return (
            f"{self.config}/{self.metric}: "
            f"{self.point_estimate:.4f} "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )


@dataclass(frozen=True)
class PairwiseResult:
    """Statistical comparison of config_b vs config_a on one metric."""

    metric: str
    config_a: str
    config_b: str
    n: int
    # Effect size (positive → config_b better)
    mean_difference: float  # mean(b) − mean(a)
    diff_ci_lower: float  # 95% bootstrap CI on the difference
    diff_ci_upper: float
    # Hypothesis test
    test: str  # "mcnemar_exact" | "permutation"
    p_value_raw: float
    p_value_adjusted: float  # Holm–Bonferroni adjusted
    alpha: float
    significant: bool  # p_value_adjusted < alpha

    def __str__(self) -> str:
        sig = "✓ significant" if self.significant else "✗ not significant"
        return (
            f"{self.config_b}−{self.config_a} / {self.metric}: "
            f"Δ={self.mean_difference:+.4f} "
            f"[{self.diff_ci_lower:+.4f}, {self.diff_ci_upper:+.4f}] "
            f"p_adj={self.p_value_adjusted:.4f} {sig}"
        )


@dataclass(frozen=True)
class AnalysisResult:
    """Complete output of one pre-registered analysis run."""

    bootstrap: tuple[BootstrapResult, ...]  # per config × metric
    pairwise: tuple[PairwiseResult, ...]  # per contrast × metric, Holm-adjusted
    b: int
    seed: int
    alpha: float


# ---------------------------------------------------------------------------
# Core statistical primitives
# ---------------------------------------------------------------------------


def bootstrap_ci(
    scores: np.ndarray,
    b: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Nonparametric bootstrap CI for the mean of *scores*.

    Resamples *scores* with replacement B times, computing the mean on each
    resample. The CI uses the percentile method.

    Args:
        scores:     1-D array of per-question metric values.
        b:          Number of bootstrap resamples.
        seed:       RNG seed for reproducibility.
        confidence: Nominal coverage (default 0.95 → 95% CI).

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    # Shape (b, n): each row is one bootstrap resample of question indices
    indices = rng.integers(0, n, size=(b, n))
    resampled_means = scores[indices].mean(axis=1)  # shape (b,)
    alpha = 1.0 - confidence
    return (
        float(scores.mean()),
        float(np.percentile(resampled_means, 100.0 * alpha / 2)),
        float(np.percentile(resampled_means, 100.0 * (1.0 - alpha / 2))),
    )


def paired_bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    b: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for the paired difference mean(b) − mean(a).

    Resamples *question indices* jointly so that each resample applies the
    same index set to both systems. This preserves the pairing and gives a
    tighter CI than treating the samples as independent.

    Args:
        scores_a, scores_b: Per-question scores, aligned by question index.
        b:                  Number of bootstrap resamples.
        seed:               RNG seed.
        confidence:         Nominal coverage.

    Returns:
        (mean_difference, ci_lower, ci_upper)
        where mean_difference = mean(b) − mean(a).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must have the same length ({len(scores_a)} vs {len(scores_b)})"
        )
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    indices = rng.integers(0, n, size=(b, n))
    # Per-resample paired difference: mean of (b_i − a_i) over resampled questions
    diffs = scores_b[indices].mean(axis=1) - scores_a[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return (
        float(scores_b.mean() - scores_a.mean()),
        float(np.percentile(diffs, 100.0 * alpha / 2)),
        float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2))),
    )


def mcnemar_exact(
    binary_a: np.ndarray,
    binary_b: np.ndarray,
) -> float:
    """Exact McNemar test for paired binary outcomes.

    Tests H₀: the probability of (A=1, B=0) equals (A=0, B=1) — i.e., neither
    system is systematically better on the questions where they disagree.

    Only the *discordant* pairs (n_10 + n_01) carry information. Under H₀,
    n_01 | (n_01 + n_10) ~ Binomial(n_01 + n_10, 0.5), giving an exact test.

    Args:
        binary_a, binary_b: Per-question 0/1 scores (e.g. exact match).

    Returns:
        Two-sided exact p-value.
    """
    a = binary_a.astype(bool)
    b = binary_b.astype(bool)
    n_10 = int(np.sum(a & ~b))  # A correct, B wrong
    n_01 = int(np.sum(~a & b))  # A wrong, B correct
    n_discordant = n_10 + n_01
    if n_discordant == 0:
        return 1.0  # Identical on every question: no evidence of a difference
    k = min(n_10, n_01)
    # Two-sided: 2 × P(X ≤ k) where X ~ Bin(n_discordant, 0.5)
    p = float(2.0 * binom.cdf(k, n_discordant, 0.5))
    return min(p, 1.0)


def paired_permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    b: int = 10_000,
    seed: int = 42,
) -> float:
    """Paired sign-flip permutation test for continuous metrics.

    Under H₀ (no difference), the sign of each per-question difference
    d_i = b_i − a_i is equally likely to be positive or negative. This test
    randomly flips signs B times and measures how often |mean of flipped diffs|
    is at least as extreme as the observed |mean(d)|.

    Args:
        scores_a, scores_b: Aligned per-question continuous scores.
        b:                  Number of permutations.
        seed:               RNG seed.

    Returns:
        Two-sided p-value.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must have the same length ({len(scores_a)} vs {len(scores_b)})"
        )
    rng = np.random.default_rng(seed)
    diffs = scores_b - scores_a
    observed = float(np.abs(diffs.mean()))
    n = len(diffs)
    # B random sign-flip patterns: shape (b, n), entries ±1
    signs = rng.choice(np.array([-1.0, 1.0]), size=(b, n))
    null_abs_means = np.abs((signs * diffs).mean(axis=1))  # shape (b,)
    return float(np.mean(null_abs_means >= observed))


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm–Bonferroni step-down correction (Holm 1979).

    Controls the family-wise error rate (FWER) at any α without assuming
    independence between tests. More powerful than Bonferroni for ordered
    rejection: tests are considered from smallest to largest raw p-value,
    and the multiplier decreases as tests are rejected.

    Algorithm:
      1. Sort tests by ascending raw p-value.
      2. For the k-th sorted test (1-indexed), multiply its p by (n − k + 1).
      3. Enforce monotonicity: each adjusted p ≥ the previous adjusted p.
      4. Cap all values at 1.

    Returns:
        Adjusted p-values in the **same order** as the input list.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, orig_idx in enumerate(order):
        # Multiplier: (n - rank) for 0-indexed rank
        raw_adj = p_values[orig_idx] * (n - rank)
        adj = float(min(1.0, max(running_max, raw_adj)))
        adjusted[orig_idx] = adj
        running_max = adj
    return adjusted


# ---------------------------------------------------------------------------
# High-level analysis
# ---------------------------------------------------------------------------


def analyse(
    data: dict[str, dict[str, np.ndarray]],
    contrasts: list[tuple[str, str]],
    metrics: list[str],
    b: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> AnalysisResult:
    """Full pre-registered analysis: bootstrap CIs + pairwise tests + Holm correction.

    Args:
        data:      Mapping {config_name: {metric_name: per_question_scores}}.
                   All score arrays for the same metric must have identical length
                   (same questions in the same order).
        contrasts: Pre-registered (config_a, config_b) pairs from EXPERIMENT.md.
        metrics:   Metric names to analyse (must exist in data for all configs).
        b:         Resamples for bootstrap and permutation tests.
        seed:      Master seed; sub-seeds are derived deterministically per test.
        alpha:     Family-wise significance level.

    Returns:
        AnalysisResult with:
          - bootstrap: one BootstrapResult per (config, metric)
          - pairwise:  one PairwiseResult per (contrast, metric), Holm-adjusted
                       within each metric across all contrasts.
    """
    # --- Per-configuration bootstrap CIs ---
    bootstrap_results: list[BootstrapResult] = []
    for config_name, metric_map in data.items():
        for metric in metrics:
            if metric not in metric_map:
                continue
            scores = metric_map[metric]
            sub = _sub_seed(seed, "boot", config_name, metric)
            est, lo, hi = bootstrap_ci(scores, b=b, seed=sub)
            bootstrap_results.append(
                BootstrapResult(
                    metric=metric,
                    config=config_name,
                    n=len(scores),
                    point_estimate=est,
                    ci_lower=lo,
                    ci_upper=hi,
                    b=b,
                )
            )

    # --- Pairwise tests: compute raw p-values for all (contrast, metric) cells ---
    # Stored as list of dicts before Holm adjustment
    raw_rows: list[dict] = []
    for metric in metrics:
        for ca, cb in contrasts:
            if metric not in data.get(ca, {}) or metric not in data.get(cb, {}):
                continue
            sa, sb = data[ca][metric], data[cb][metric]
            n = len(sa)
            # Paired bootstrap CI on the difference
            sub_d = _sub_seed(seed, "diff", ca, cb, metric)
            mean_diff, diff_lo, diff_hi = paired_bootstrap_ci(sa, sb, b=b, seed=sub_d)
            # Choose test by whether the metric is binary (EM) or continuous (F1)
            is_binary = set(np.unique(sa)).issubset({0.0, 1.0}) and set(np.unique(sb)).issubset(
                {0.0, 1.0}
            )
            if is_binary:
                p_raw = mcnemar_exact(sa, sb)
                test = "mcnemar_exact"
            else:
                sub_t = _sub_seed(seed, "perm", ca, cb, metric)
                p_raw = paired_permutation_test(sa, sb, b=b, seed=sub_t)
                test = "permutation"
            raw_rows.append(
                dict(
                    metric=metric,
                    config_a=ca,
                    config_b=cb,
                    n=n,
                    mean_difference=mean_diff,
                    diff_ci_lower=diff_lo,
                    diff_ci_upper=diff_hi,
                    test=test,
                    p_value_raw=p_raw,
                )
            )

    # --- Holm–Bonferroni: adjust within each metric across contrasts ---
    pairwise_results: list[PairwiseResult] = []
    for metric in metrics:
        metric_rows = [r for r in raw_rows if r["metric"] == metric]
        if not metric_rows:
            continue
        raw_ps = [r["p_value_raw"] for r in metric_rows]
        adj_ps = holm_bonferroni(raw_ps)
        for row, p_adj in zip(metric_rows, adj_ps):
            pairwise_results.append(
                PairwiseResult(
                    metric=row["metric"],
                    config_a=row["config_a"],
                    config_b=row["config_b"],
                    n=row["n"],
                    mean_difference=row["mean_difference"],
                    diff_ci_lower=row["diff_ci_lower"],
                    diff_ci_upper=row["diff_ci_upper"],
                    test=row["test"],
                    p_value_raw=row["p_value_raw"],
                    p_value_adjusted=p_adj,
                    alpha=alpha,
                    significant=p_adj < alpha,
                )
            )

    return AnalysisResult(
        bootstrap=tuple(bootstrap_results),
        pairwise=tuple(pairwise_results),
        b=b,
        seed=seed,
        alpha=alpha,
    )

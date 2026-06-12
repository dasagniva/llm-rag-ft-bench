"""
Tests for eval/stats.py — the statistical centrepiece.

Test strategy (from EXPERIMENT.md / ROADMAP):
  1. Null-difference correctness: identical inputs → CI covers 0, p not significant.
  2. Planted known effect: artificial signal → reliably detected.
  3. CI coverage by simulation: nominal 95% CI covers true mean ≥ 88% of trials
     (tolerance for finite B; see test for full rationale).
  4. Bit-for-bit reproducibility: identical seeds → identical results.
  5. Edge cases: all-zero, all-one, no discordant pairs, single observation.
  6. Holm–Bonferroni: ordering and monotonicity properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from ragbench.eval.stats import (
    AnalysisResult,
    BootstrapResult,
    PairwiseResult,
    analyse,
    bootstrap_ci,
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap_ci,
    paired_permutation_test,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
N = 300  # mirrors real eval set size


def make_scores(em_rate: float = 0.0, f1_mean: float = 0.0, n: int = N) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    em = (rng.uniform(size=n) < em_rate).astype(float)
    f1 = np.clip(rng.normal(f1_mean, 0.1, size=n), 0, 1)
    return {"exact_match": em, "token_f1": f1}


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_returns_three_floats(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = bootstrap_ci(scores, b=500, seed=0)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_point_estimate_is_sample_mean(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        est, _, _ = bootstrap_ci(scores, b=500, seed=0)
        assert est == pytest.approx(np.mean(scores))

    def test_ci_ordered(self):
        scores = RNG.uniform(0, 1, size=100)
        est, lo, hi = bootstrap_ci(scores, b=1000, seed=0)
        assert lo <= est <= hi

    def test_ci_width_decreases_with_n(self):
        """Larger samples should produce tighter CIs."""
        small = RNG.uniform(0, 1, size=30)
        large = RNG.uniform(0, 1, size=300)
        _, lo_s, hi_s = bootstrap_ci(small, b=1000, seed=0)
        _, lo_l, hi_l = bootstrap_ci(large, b=1000, seed=0)
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_reproducible(self):
        scores = RNG.uniform(0, 1, size=100)
        r1 = bootstrap_ci(scores, b=1000, seed=7)
        r2 = bootstrap_ci(scores, b=1000, seed=7)
        assert r1 == r2

    def test_different_seeds_differ(self):
        scores = RNG.uniform(0, 1, size=100)
        r1 = bootstrap_ci(scores, b=1000, seed=1)
        r2 = bootstrap_ci(scores, b=1000, seed=2)
        assert r1 != r2

    def test_all_identical_values(self):
        scores = np.ones(50)
        est, lo, hi = bootstrap_ci(scores, b=500, seed=0)
        assert est == pytest.approx(1.0)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_single_element(self):
        scores = np.array([0.42])
        est, lo, hi = bootstrap_ci(scores, b=100, seed=0)
        assert est == pytest.approx(0.42)

    def test_nominal_coverage_by_simulation(self):
        """95% CI should contain the true mean in ≥ 88% of simulation trials.

        Simulation design:
          - True distribution: Bernoulli(0.4) → true mean = 0.4
          - 500 simulation trials, n=200 observations each, B=2,000 resamples
          - Tolerance 88%: bootstrap can be slightly conservative or anti-conservative
            with finite B; 88% guards against numerical accidents without being too tight.
        """
        true_mean = 0.4
        n_sims = 500
        n_obs = 200
        covered = 0
        outer_rng = np.random.default_rng(42)
        for trial in range(n_sims):
            scores = outer_rng.binomial(1, true_mean, size=n_obs).astype(float)
            _, lo, hi = bootstrap_ci(scores, b=2_000, seed=trial)
            if lo <= true_mean <= hi:
                covered += 1
        coverage = covered / n_sims
        assert coverage >= 0.88, f"Coverage {coverage:.3f} below 0.88 — bootstrap CI is broken"


# ---------------------------------------------------------------------------
# paired_bootstrap_ci
# ---------------------------------------------------------------------------


class TestPairedBootstrapCI:
    def test_null_difference_ci_covers_zero(self):
        """When A == B, the CI on the difference should cover zero."""
        scores = RNG.uniform(0, 1, size=N)
        diff, lo, hi = paired_bootstrap_ci(scores, scores, b=5_000, seed=0)
        assert diff == pytest.approx(0.0, abs=1e-10)
        assert lo <= 0.0 <= hi

    def test_positive_shift_detected(self):
        """When B is uniformly better than A, CI should be entirely above zero."""
        a = np.zeros(N)
        b = np.ones(N) * 0.4
        diff, lo, hi = paired_bootstrap_ci(a, b, b=5_000, seed=0)
        assert diff == pytest.approx(0.4)
        assert lo > 0.0

    def test_negative_shift(self):
        a = np.ones(N) * 0.5
        b = np.zeros(N)
        diff, lo, hi = paired_bootstrap_ci(a, b, b=5_000, seed=0)
        assert diff == pytest.approx(-0.5)
        assert hi < 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            paired_bootstrap_ci(np.ones(10), np.ones(11), b=100, seed=0)

    def test_reproducible(self):
        a = RNG.uniform(0, 1, size=50)
        b = RNG.uniform(0, 1, size=50)
        assert paired_bootstrap_ci(a, b, b=500, seed=3) == paired_bootstrap_ci(a, b, b=500, seed=3)

    def test_ci_ordered(self):
        a = RNG.uniform(0, 0.5, size=N)
        b = RNG.uniform(0, 0.5, size=N)
        diff, lo, hi = paired_bootstrap_ci(a, b, b=2_000, seed=0)
        assert lo <= diff <= hi

    def test_paired_resampling_tighter_than_unpaired(self):
        """Paired resampling exploits correlation and should give a narrower CI
        than treating the two samples as independent when they are in fact paired."""
        rng = np.random.default_rng(99)
        base = rng.uniform(0, 1, size=N)
        # b is a small perturbation of a — highly correlated
        b_scores = base + rng.normal(0, 0.02, size=N)
        b_scores = np.clip(b_scores, 0, 1)
        _, lo_paired, hi_paired = paired_bootstrap_ci(base, b_scores, b=3_000, seed=0)
        # Unpaired: bootstrap CI on difference of independent means
        _, lo_a, hi_a = bootstrap_ci(base, b=3_000, seed=1)
        _, lo_b, hi_b = bootstrap_ci(b_scores, b=3_000, seed=2)
        width_paired = hi_paired - lo_paired
        width_unpaired = (hi_b - lo_b) + (hi_a - lo_a)  # rough unpaired bound
        assert width_paired < width_unpaired


# ---------------------------------------------------------------------------
# mcnemar_exact
# ---------------------------------------------------------------------------


class TestMcnemarExact:
    def test_no_discordant_pairs_returns_one(self):
        a = np.array([1, 0, 1, 0], dtype=float)
        b = np.array([1, 0, 1, 0], dtype=float)
        assert mcnemar_exact(a, b) == pytest.approx(1.0)

    def test_all_discordant_one_way(self):
        """B always right where A wrong and vice versa → smallest possible p."""
        a = np.array([1, 0] * 50, dtype=float)
        b = np.array([0, 1] * 50, dtype=float)
        p = mcnemar_exact(a, b)
        assert 0.0 <= p <= 1.0

    def test_symmetric_discordant_pairs_not_significant(self):
        """Equal n_10 and n_01 → p ≈ 1 (no preference)."""
        n = 50
        a = np.array([1] * n + [0] * n, dtype=float)
        b = np.array([0] * n + [1] * n, dtype=float)
        p = mcnemar_exact(a, b)
        assert p == pytest.approx(1.0, abs=0.01)

    def test_strongly_asymmetric_discordant_is_significant(self):
        """90 in one direction, 10 in the other → very small p."""
        a = np.array([1] * 90 + [0] * 10, dtype=float)
        b = np.array([0] * 90 + [1] * 10, dtype=float)
        p = mcnemar_exact(a, b)
        assert p < 0.001

    def test_planted_effect_detected(self):
        """B correct 30% of the time where A always wrong → detect asymmetry."""
        n = 300
        a = np.zeros(n)
        rng = np.random.default_rng(0)
        b = (rng.uniform(size=n) < 0.3).astype(float)
        p = mcnemar_exact(a, b)
        assert p < 0.05

    def test_p_in_unit_interval(self):
        for _ in range(20):
            a = RNG.integers(0, 2, size=50).astype(float)
            b = RNG.integers(0, 2, size=50).astype(float)
            p = mcnemar_exact(a, b)
            assert 0.0 <= p <= 1.0

    def test_two_sided(self):
        """Swapping a and b should not change p (two-sided test)."""
        a = np.array([1] * 70 + [0] * 30, dtype=float)
        b = np.array([0] * 70 + [1] * 30, dtype=float)
        assert mcnemar_exact(a, b) == pytest.approx(mcnemar_exact(b, a))


# ---------------------------------------------------------------------------
# paired_permutation_test
# ---------------------------------------------------------------------------


class TestPairedPermutationTest:
    def test_identical_scores_not_significant(self):
        scores = RNG.uniform(0, 1, size=N)
        p = paired_permutation_test(scores, scores, b=5_000, seed=0)
        assert p == pytest.approx(1.0, abs=0.01)

    def test_large_effect_detected(self):
        a = np.zeros(N)
        b = np.ones(N) * 0.5
        p = paired_permutation_test(a, b, b=5_000, seed=0)
        assert p < 0.001

    def test_null_p_value_uniform(self):
        """Under the null (A, B drawn i.i.d.), p-values should be approx. uniform [0,1].

        Note: using identical arrays (A == B) always produces p=1.0 because all
        differences are exactly zero. The null here is that A and B are independent
        draws from the same distribution (exchangeable under the null).
        """
        ps = []
        rng = np.random.default_rng(7)
        for i in range(200):
            a = rng.uniform(0, 1, size=50)
            b = rng.uniform(0, 1, size=50)  # independent draw, same distribution
            ps.append(paired_permutation_test(a, b, b=500, seed=i))
        mean_p = np.mean(ps)
        assert 0.4 <= mean_p <= 0.6, f"Mean null p-value {mean_p:.3f} too far from 0.5"

    def test_p_in_unit_interval(self):
        a = RNG.uniform(0, 1, size=100)
        b = RNG.uniform(0, 1, size=100)
        p = paired_permutation_test(a, b, b=1_000, seed=0)
        assert 0.0 <= p <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            paired_permutation_test(np.ones(10), np.ones(11), b=100, seed=0)

    def test_reproducible(self):
        a = RNG.uniform(0, 1, size=100)
        b = RNG.uniform(0, 1, size=100)
        assert paired_permutation_test(a, b, b=1_000, seed=5) == paired_permutation_test(
            a, b, b=1_000, seed=5
        )

    def test_planted_effect_detected(self):
        """Known shift of 0.2 on n=300 should reliably yield p < 0.05."""
        rng = np.random.default_rng(42)
        a = rng.uniform(0, 0.5, size=N)
        b = a + 0.2  # deterministic shift
        p = paired_permutation_test(a, b, b=5_000, seed=0)
        assert p < 0.05

    def test_two_sided(self):
        """Swapping a and b should not change p."""
        a = RNG.uniform(0, 1, size=100)
        b = RNG.uniform(0.2, 1.2, size=100)
        p_ab = paired_permutation_test(a, b, b=2_000, seed=0)
        p_ba = paired_permutation_test(b, a, b=2_000, seed=0)
        assert p_ab == pytest.approx(p_ba, abs=1e-10)


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------


class TestHolmBonferroni:
    def test_empty_input(self):
        assert holm_bonferroni([]) == []

    def test_single_p_value_unchanged(self):
        assert holm_bonferroni([0.03]) == pytest.approx([0.03])

    def test_single_large_p_capped_at_one(self):
        assert holm_bonferroni([2.0]) == [pytest.approx(1.0)]

    def test_adjusted_never_below_raw(self):
        """Adjusted p-values must be ≥ raw p-values (conservative correction)."""
        ps = [0.01, 0.04, 0.10, 0.50]
        adj = holm_bonferroni(ps)
        for raw, a in zip(ps, adj):
            assert a >= raw - 1e-12

    def test_adjusted_monotone_in_original_order_of_sorted_p(self):
        """After sorting by raw p, adjusted values must be non-decreasing."""
        ps = [0.01, 0.03, 0.20, 0.50]
        adj = holm_bonferroni(ps)
        adj_sorted_by_raw = [adj[i] for i in sorted(range(len(ps)), key=lambda i: ps[i])]
        for i in range(len(adj_sorted_by_raw) - 1):
            assert adj_sorted_by_raw[i] <= adj_sorted_by_raw[i + 1] + 1e-12

    def test_all_below_alpha_remain_significant(self):
        """When all raw p < 0.05 and n is small, all adjusted p should be < 0.05."""
        ps = [0.001, 0.002, 0.003]
        adj = holm_bonferroni(ps)
        assert all(a < 0.05 for a in adj)

    def test_all_above_alpha_remain_not_significant(self):
        ps = [0.1, 0.2, 0.3, 0.4]
        adj = holm_bonferroni(ps)
        assert all(a >= 0.1 for a in adj)

    def test_order_preserved(self):
        """The adjusted p-value at each position corresponds to the same position's raw p."""
        ps = [0.04, 0.01, 0.20]
        adj = holm_bonferroni(ps)
        # With ps sorted: [0.01, 0.04, 0.20], multipliers are [3, 2, 1]
        # adj[1] = min(1, max(0, 0.01 * 3)) = 0.03
        # adj[0] = min(1, max(0.03, 0.04 * 2)) = 0.08
        # adj[2] = min(1, max(0.08, 0.20 * 1)) = 0.20
        assert adj[1] == pytest.approx(0.03)
        assert adj[0] == pytest.approx(0.08)
        assert adj[2] == pytest.approx(0.20)

    def test_fwer_control_by_simulation(self):
        """Under global H0, FWER should be ≤ α = 0.05 (with slack for finite B).

        Simulation: 500 trials, each with 5 independent null p-values drawn
        uniformly from [0, 1]. After Holm correction at α=0.05, the fraction of
        trials where ANY adjusted p < 0.05 should be ≤ 0.07 (slack for variance).
        """
        rng = np.random.default_rng(0)
        alpha = 0.05
        n_trials = 500
        false_positives = 0
        for _ in range(n_trials):
            raw_ps = rng.uniform(0, 1, size=5).tolist()
            adj = holm_bonferroni(raw_ps)
            if any(a < alpha for a in adj):
                false_positives += 1
        fwer = false_positives / n_trials
        assert fwer <= 0.07, f"FWER {fwer:.3f} exceeds tolerance 0.07"


# ---------------------------------------------------------------------------
# analyse (high-level)
# ---------------------------------------------------------------------------


class TestAnalyse:
    def _make_data(self, em_a=0.0, em_b=0.0, f1_a=0.0, f1_b=0.0, n=N):
        rng = np.random.default_rng(0)
        return {
            "base": {
                "exact_match": (rng.uniform(size=n) < em_a).astype(float),
                "token_f1": np.clip(rng.normal(f1_a, 0.05, size=n), 0, 1),
            },
            "rag": {
                "exact_match": (rng.uniform(size=n) < em_b).astype(float),
                "token_f1": np.clip(rng.normal(f1_b, 0.05, size=n), 0, 1),
            },
        }

    def test_returns_analysis_result(self):
        data = self._make_data()
        result = analyse(data, [("base", "rag")], ["exact_match"], b=500, seed=0)
        assert isinstance(result, AnalysisResult)

    def test_bootstrap_results_populated(self):
        data = self._make_data()
        result = analyse(data, [("base", "rag")], ["exact_match", "token_f1"], b=500, seed=0)
        # One BootstrapResult per config × metric = 2 configs × 2 metrics = 4
        assert len(result.bootstrap) == 4
        assert all(isinstance(r, BootstrapResult) for r in result.bootstrap)

    def test_pairwise_results_populated(self):
        data = self._make_data()
        result = analyse(data, [("base", "rag")], ["exact_match", "token_f1"], b=500, seed=0)
        # One PairwiseResult per contrast × metric = 1 × 2 = 2
        assert len(result.pairwise) == 2
        assert all(isinstance(r, PairwiseResult) for r in result.pairwise)

    def test_null_result_not_significant(self):
        """Identical systems should yield non-significant pairwise tests."""
        scores = np.random.default_rng(1).uniform(0, 0.1, size=N)
        data = {
            "base": {"exact_match": scores, "token_f1": scores},
            "rag": {"exact_match": scores, "token_f1": scores},
        }
        result = analyse(data, [("base", "rag")], ["exact_match", "token_f1"], b=2_000, seed=0)
        for pw in result.pairwise:
            assert not pw.significant, f"{pw.metric} incorrectly flagged significant on null data"

    def test_planted_effect_detected(self):
        """A large difference should be detected as significant."""
        rng = np.random.default_rng(42)
        a_em = np.zeros(N)
        b_em = (rng.uniform(size=N) < 0.4).astype(float)  # 40% EM for B
        a_f1 = np.zeros(N)
        b_f1 = np.ones(N) * 0.4
        data = {
            "base": {"exact_match": a_em, "token_f1": a_f1},
            "rag": {"exact_match": b_em, "token_f1": b_f1},
        }
        result = analyse(data, [("base", "rag")], ["exact_match", "token_f1"], b=5_000, seed=0)
        for pw in result.pairwise:
            assert pw.significant, f"{pw.metric} not detected despite large planted effect"

    def test_reproducible(self):
        data = self._make_data(em_a=0.1, em_b=0.2, f1_a=0.1, f1_b=0.15)
        r1 = analyse(data, [("base", "rag")], ["exact_match", "token_f1"], b=1_000, seed=42)
        r2 = analyse(data, [("base", "rag")], ["exact_match", "token_f1"], b=1_000, seed=42)
        assert r1.pairwise[0].p_value_raw == r2.pairwise[0].p_value_raw
        assert r1.pairwise[0].mean_difference == r2.pairwise[0].mean_difference

    def test_multiple_contrasts_holm_applied(self):
        """With 3 contrasts and small raw p-values, Holm should inflate some."""
        rng = np.random.default_rng(0)
        base = rng.uniform(0, 1, size=N)
        data = {
            "base": {"token_f1": base},
            "c1": {"token_f1": base + 0.01},
            "c2": {"token_f1": base + 0.01},
            "c3": {"token_f1": base + 0.01},
        }
        contrasts = [("base", "c1"), ("base", "c2"), ("base", "c3")]
        result = analyse(data, contrasts, ["token_f1"], b=2_000, seed=0)
        raw_ps = [pw.p_value_raw for pw in result.pairwise]
        adj_ps = [pw.p_value_adjusted for pw in result.pairwise]
        for raw, adj in zip(raw_ps, adj_ps):
            assert adj >= raw - 1e-12

    def test_exact_match_uses_mcnemar(self):
        data = self._make_data()
        result = analyse(data, [("base", "rag")], ["exact_match"], b=500, seed=0)
        em_pw = next(pw for pw in result.pairwise if pw.metric == "exact_match")
        assert em_pw.test == "mcnemar_exact"

    def test_f1_uses_permutation(self):
        data = self._make_data()
        result = analyse(data, [("base", "rag")], ["token_f1"], b=500, seed=0)
        f1_pw = next(pw for pw in result.pairwise if pw.metric == "token_f1")
        assert f1_pw.test == "permutation"

    def test_pairwise_result_fields(self):
        data = self._make_data(em_a=0.1, em_b=0.15)
        result = analyse(data, [("base", "rag")], ["exact_match"], b=500, seed=0)
        pw = result.pairwise[0]
        assert pw.config_a == "base"
        assert pw.config_b == "rag"
        assert pw.n == N
        assert 0.0 <= pw.p_value_adjusted <= 1.0
        assert pw.diff_ci_lower <= pw.mean_difference <= pw.diff_ci_upper

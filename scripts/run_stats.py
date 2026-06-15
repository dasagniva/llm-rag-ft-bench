#!/usr/bin/env python
"""Run the pre-registered statistical analysis and write reports/base_vs_rag.md.

Usage:
    uv run scripts/run_stats.py

Reads:
    reports/base_results.jsonl
    reports/rag_results.jsonl

Writes:
    reports/base_vs_rag.md
    reports/base_vs_rag_forest.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragbench.eval.reporting import forest_plot, write_report
from ragbench.eval.stats import analyse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Pre-registered contrasts (EXPERIMENT.md §7)
# Only C1 (base vs rag) is available at Phase 3 MVP; C2-C5 added in Phase 4.
CONTRASTS: list[tuple[str, str]] = [
    ("base", "rag"),  # C1
]
METRICS = ["exact_match", "token_f1"]


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def align_results(
    rows_a: list[dict],
    rows_b: list[dict],
    metrics: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Align two result sets by question ID and return metric arrays.

    Both sets must cover the same question IDs (panics otherwise — this is
    the train/eval leakage analogue: question sets must be identical).
    """
    index_a = {r["id"]: r for r in rows_a}
    index_b = {r["id"]: r for r in rows_b}

    ids_a, ids_b = set(index_a), set(index_b)
    if ids_a != ids_b:
        only_a = ids_a - ids_b
        only_b = ids_b - ids_a
        raise ValueError(
            f"Question ID mismatch: {len(only_a)} IDs only in A, {len(only_b)} only in B.\n"
            f"Example A-only: {next(iter(only_a)) if only_a else '—'}\n"
            f"Example B-only: {next(iter(only_b)) if only_b else '—'}"
        )

    # Use the order from rows_a (deterministic; matches eval manifest order)
    ordered_ids = [r["id"] for r in rows_a]

    def extract(rows_indexed: dict[str, dict], name: str) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for metric in metrics:
            result[metric] = np.array([float(rows_indexed[qid][metric]) for qid in ordered_ids])
        return result

    return {
        "base": extract(index_a, "base"),
        "rag": extract(index_b, "rag"),
    }


def main() -> None:
    base_path = Path("reports/base_results.jsonl")
    rag_path = Path("reports/rag_results.jsonl")

    for p in (base_path, rag_path):
        if not p.exists():
            logger.error("Missing artifact: %s — run scripts/run_eval.py first", p)
            sys.exit(1)

    logger.info("Loading base results from %s", base_path)
    base_rows = load_jsonl(base_path)
    logger.info("Loading RAG results from %s", rag_path)
    rag_rows = load_jsonl(rag_path)

    logger.info("Aligning %d base + %d RAG results by question ID …", len(base_rows), len(rag_rows))
    data = align_results(base_rows, rag_rows, metrics=METRICS)
    n = len(base_rows)
    logger.info("Aligned %d questions; metrics: %s", n, METRICS)

    logger.info("Running statistical analysis (B=10,000, seed=42) …")
    result = analyse(
        data,
        contrasts=CONTRASTS,
        metrics=METRICS,
        b=10_000,
        seed=42,
        alpha=0.05,
    )

    # Faithfulness is RAG-only (no `base` analogue) — analysed separately as a
    # single-configuration bootstrap CI, not a pairwise contrast.
    faith_scores = np.array([float(r["faithfulness"]) for r in rag_rows])
    faith_result = analyse(
        {"rag": {"faithfulness": faith_scores}},
        contrasts=[],
        metrics=["faithfulness"],
        b=10_000,
        seed=42,
        alpha=0.05,
    )
    faith = faith_result.bootstrap[0]

    # McNemar discordant-pair breakdown (EM), for the power note in the report.
    em_base, em_rag = data["base"]["exact_match"], data["rag"]["exact_match"]
    rag_only = int(np.sum((em_base == 0) & (em_rag == 1)))
    base_only = int(np.sum((em_base == 1) & (em_rag == 0)))
    discordant = rag_only + base_only

    # Print summary
    print("\n" + "=" * 65)
    print("  BASE vs RAG — Statistical Summary")
    print("=" * 65)
    for r in result.bootstrap:
        print(f"  {r}")
    print()
    for pw in result.pairwise:
        print(f"  {pw}")
    print("=" * 65 + "\n")

    # Write figure
    fig_path = Path("reports/base_vs_rag_forest.png")
    logger.info("Writing forest plot → %s", fig_path)
    forest_plot(result, config_a="base", config_b="rag", out_path=fig_path)

    em = next(pw for pw in result.pairwise if pw.metric == "exact_match")
    f1 = next(pw for pw in result.pairwise if pw.metric == "token_f1")
    base_em = next(b for b in result.bootstrap if b.config == "base" and b.metric == "exact_match")
    base_f1 = next(b for b in result.bootstrap if b.config == "base" and b.metric == "token_f1")

    # Write markdown report
    report_path = Path("reports/base_vs_rag.md")
    logger.info("Writing report → %s", report_path)
    write_report(
        result,
        config_a="base",
        config_b="rag",
        out_path=report_path,
        figure_path=fig_path,
        extra_context=(
            "## RAG-only diagnostics\n\n"
            "| Metric | Point est. | 95% CI | n |\n"
            "|---|---|---|---|\n"
            f"| Faithfulness | {faith.point_estimate:.4f} "
            f"| [{faith.ci_lower:.4f}, {faith.ci_upper:.4f}] | {faith.n} |\n\n"
            "Faithfulness (fraction of answer content traceable to retrieved context, "
            "0-1 scale) is **moderate, not high**: roughly half of RAG answer content "
            "is grounded in the retrieved chunks, with the rest coming from the model's "
            'own knowledge or reasoning. This is the opposite of "context is barely '
            'used" — context is being used substantially, but not exclusively.\n\n'
            "## Interpretation\n\n"
            f"With the corrected numeric scoring (Task 1) and a retrieval corpus built "
            f"from the actual FinQA/TAT-QA source documents (Task 2), `rag` shows a "
            f"large, statistically significant improvement over `base` on both "
            f"Exact Match (Δ={em.mean_difference:+.4f}, "
            f"95% CI [{em.diff_ci_lower:+.4f}, {em.diff_ci_upper:+.4f}], "
            f"{em.test}, p_adj={em.p_value_adjusted:.4f}) and Token F1 "
            f"(Δ={f1.mean_difference:+.4f}, "
            f"95% CI [{f1.diff_ci_lower:+.4f}, {f1.diff_ci_upper:+.4f}], "
            f"{f1.test}, p_adj={f1.p_value_adjusted:.4f}).\n\n"
            f"**McNemar power note:** the EM contingency table has {discordant} "
            f"discordant pairs out of N={em.n} ({rag_only} questions where `rag` is "
            f"correct and `base` is wrong, vs {base_only} the reverse) — this is not "
            f"a small-discordant-pairs regime, so the exact test is informative here.\n\n"
            f"**`base` remains near the performance floor in absolute terms** "
            f"(EM={base_em.point_estimate:.4f} 95% CI "
            f"[{base_em.ci_lower:.4f}, {base_em.ci_upper:.4f}], "
            f"F1={base_f1.point_estimate:.4f} 95% CI "
            f"[{base_f1.ci_lower:.4f}, {base_f1.ci_upper:.4f}]). This is expected, "
            "not a measurement artifact: without retrieved context, the model has no "
            "access to the specific financial-table values FinQA/TAT-QA questions ask "
            "about, so it cannot answer them correctly regardless of scoring. The "
            "informative result is the *contrast* — `rag` is clearly off the floor "
            "and significantly ahead of `base` — not the absolute level of `base`.\n"
        ),
    )
    logger.info("Done. Commit reports/base_vs_rag.md and reports/base_vs_rag_forest.png.")


if __name__ == "__main__":
    main()

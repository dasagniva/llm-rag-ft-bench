"""
Generate reports and figures from AnalysisResult.

Functions here perform I/O (writing files, plotting); they are intentionally
kept separate from the pure statistical functions in stats.py.
"""

from __future__ import annotations

from pathlib import Path

from ragbench.eval.stats import AnalysisResult

# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------


def forest_plot(
    result: AnalysisResult,
    config_a: str,
    config_b: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    """Save a forest-plot figure of pairwise metric differences with 95% CIs.

    Each row shows one metric: the point estimate of mean(b) − mean(a) as a
    circle, the 95% bootstrap CI as a horizontal bar, and a vertical reference
    line at zero. Points are coloured green (significant) or red (not significant)
    according to the Holm-adjusted p-value.

    Args:
        result:   AnalysisResult from stats.analyse().
        config_a: Name of the reference configuration (left side of comparison).
        config_b: Name of the test configuration (right side).
        out_path: Where to save the figure (.png or .pdf).
        title:    Figure title; defaults to "config_b − config_a".
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    pairwise = [pw for pw in result.pairwise if pw.config_a == config_a and pw.config_b == config_b]
    if not pairwise:
        raise ValueError(f"No pairwise results found for ({config_a}, {config_b})")

    n_rows = len(pairwise)
    fig_height = max(3.0, n_rows * 1.4 + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    if title is None:
        title = (
            f"Metric differences: {config_b} − {config_a}\nwith 95% bootstrap CI (B={result.b:,})"
        )

    y_positions = list(range(n_rows))
    metric_labels = []

    for i, pw in enumerate(pairwise):
        color = "tab:green" if pw.significant else "tab:red"
        # Error bar lengths (distance from point to CI bound)
        xerr_lo = pw.mean_difference - pw.diff_ci_lower
        xerr_hi = pw.diff_ci_upper - pw.mean_difference

        ax.errorbar(
            pw.mean_difference,
            i,
            xerr=[[xerr_lo], [xerr_hi]],
            fmt="o",
            color=color,
            capsize=6,
            markersize=9,
            linewidth=2,
            label="_nolegend_",
        )

        # Annotation: Δ, CI, p-value
        sig_marker = "✓" if pw.significant else "✗"
        annotation = (
            f" Δ={pw.mean_difference:+.4f}  "
            f"[{pw.diff_ci_lower:+.4f}, {pw.diff_ci_upper:+.4f}]\n"
            f" p_adj={pw.p_value_adjusted:.3f} ({pw.test}) {sig_marker}"
        )
        ax.annotate(
            annotation,
            xy=(pw.mean_difference, i),
            xytext=(12, 0),
            textcoords="offset points",
            fontsize=8,
            va="center",
            color=color,
        )

        label = pw.metric.replace("_", " ").title()
        metric_labels.append(label)

    # Reference line at zero
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5, zorder=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(metric_labels, fontsize=11)
    ax.set_xlabel(f"Difference ({config_b} − {config_a})", fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Legend for significance
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:green",
            markersize=9,
            label=f"Significant (p_adj < {result.alpha})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:red",
            markersize=9,
            label="Not significant",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.set_ylim(-0.8, n_rows - 0.2)
    ax.invert_yaxis()  # top metric at top of plot
    ax.grid(axis="x", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown results table
# ---------------------------------------------------------------------------


def results_table_md(
    result: AnalysisResult,
    config_a: str,
    config_b: str,
) -> str:
    """Generate a markdown results table for one contrast.

    Returns a string containing the per-configuration bootstrap CI table and
    the pairwise comparison table, ready to paste into a report.
    """
    bootstrap_a = {r.metric: r for r in result.bootstrap if r.config == config_a}
    bootstrap_b = {r.metric: r for r in result.bootstrap if r.config == config_b}
    pairwise = {
        pw.metric: pw
        for pw in result.pairwise
        if pw.config_a == config_a and pw.config_b == config_b
    }
    metrics = list(pairwise.keys())

    lines: list[str] = []

    # --- Per-configuration table ---
    lines.append(f"### Per-configuration results (B={result.b:,}, seed={result.seed})")
    lines.append("")
    lines.append(
        "| Metric | "
        f"{config_a} point est. | {config_a} 95% CI | "
        f"{config_b} point est. | {config_b} 95% CI |"
    )
    lines.append("|---|---|---|---|---|")
    for metric in metrics:
        ba = bootstrap_a.get(metric)
        bb = bootstrap_b.get(metric)
        label = metric.replace("_", " ").title()
        a_est = f"{ba.point_estimate:.4f}" if ba else "—"
        a_ci = f"[{ba.ci_lower:.4f}, {ba.ci_upper:.4f}]" if ba else "—"
        b_est = f"{bb.point_estimate:.4f}" if bb else "—"
        b_ci = f"[{bb.ci_lower:.4f}, {bb.ci_upper:.4f}]" if bb else "—"
        lines.append(f"| {label} | {a_est} | {a_ci} | {b_est} | {b_ci} |")

    lines.append("")

    # --- Pairwise comparison table ---
    lines.append(f"### Pairwise comparison: {config_b} − {config_a} (α={result.alpha})")
    lines.append("")
    lines.append(
        "| Metric | Δ (mean diff) | 95% CI on Δ | Test | p (raw) | p (adj, Holm) | Significant? |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for metric in metrics:
        pw = pairwise.get(metric)
        if pw is None:
            continue
        label = metric.replace("_", " ").title()
        diff = f"{pw.mean_difference:+.4f}"
        ci = f"[{pw.diff_ci_lower:+.4f}, {pw.diff_ci_upper:+.4f}]"
        test = pw.test.replace("_", " ")
        p_raw = f"{pw.p_value_raw:.4f}"
        p_adj = f"{pw.p_value_adjusted:.4f}"
        sig = "✓ Yes" if pw.significant else "✗ No"
        lines.append(f"| {label} | {diff} | {ci} | {test} | {p_raw} | {p_adj} | {sig} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full markdown report
# ---------------------------------------------------------------------------


def write_report(
    result: AnalysisResult,
    config_a: str,
    config_b: str,
    out_path: Path,
    figure_path: Path,
    extra_context: str = "",
) -> None:
    """Write a complete markdown report to *out_path*.

    Args:
        result:       AnalysisResult from stats.analyse().
        config_a:     Reference configuration name.
        config_b:     Test configuration name.
        out_path:     Destination .md file.
        figure_path:  Path to the already-saved forest plot figure.
        extra_context: Optional additional markdown to append (methodology notes, etc.)
    """
    pairwise = [pw for pw in result.pairwise if pw.config_a == config_a and pw.config_b == config_b]

    # One-sentence finding
    any_sig = any(pw.significant for pw in pairwise)
    if any_sig:
        sig_metrics = [pw.metric.replace("_", " ") for pw in pairwise if pw.significant]
        finding = (
            f"**{config_b.upper()} shows a statistically significant improvement over "
            f"{config_a.upper()} on: {', '.join(sig_metrics)}** "
            f"(Holm-adjusted α={result.alpha})."
        )
    else:
        diffs = ", ".join(
            f"{pw.metric.replace('_', ' ')} Δ={pw.mean_difference:+.4f} "
            f"[{pw.diff_ci_lower:+.4f}, {pw.diff_ci_upper:+.4f}]"
            for pw in pairwise
        )
        finding = (
            f"**No statistically significant difference detected between "
            f"{config_b.upper()} and {config_a.upper()}** "
            f"at the Holm-adjusted α={result.alpha} level. "
            f"Effect sizes and CIs: {diffs}."
        )

    fig_rel = figure_path.name  # relative link within reports/
    lines = [
        f"# Statistical comparison: {config_b} vs {config_a}",
        "",
        "*Generated by `scripts/run_stats.py` — numbers come from logged MLflow runs.*",
        "",
        "## Headline finding",
        "",
        finding,
        "",
        f"![Forest plot]({fig_rel})",
        "",
        "## Results",
        "",
        results_table_md(result, config_a, config_b),
        "",
        "## Statistical methodology",
        "",
        f"- **Bootstrap CIs:** B={result.b:,} paired resamples of question indices "
        f"(seed={result.seed}); percentile method; 95% confidence.",
        "- **McNemar exact test:** applied to binary Exact Match outcomes.",
        "- **Paired permutation test:** sign-flip permutation on per-question F1 "
        f"differences; B={result.b:,} permutations.",
        "- **Multiple-comparison correction:** Holm–Bonferroni across all "
        "pre-registered contrasts within each metric (see `EXPERIMENT.md`).",
        "",
    ]
    if extra_context:
        lines.append(extra_context)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))

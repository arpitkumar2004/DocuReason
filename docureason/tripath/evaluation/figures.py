"""figures.py - publication figures and a derived-insight digest from the reports.

Turns ``eval_suite.json`` and ``ablation_report.json`` into the small set of figures the
numbers actually support, plus an ``INSIGHTS.md`` whose every sentence is computed from
those files. Nothing here invents a series: a figure whose inputs are missing is skipped
and named in the digest, because an empty axis reads as "measured, and flat".

Design constraints, held deliberately:

* **One measure per axis.** Latency and [0, 1] quality metrics never share a chart - a
  second y-scale is the one chart mistake that makes every comparison on it unreadable.
* **Intervals, not points.** Every estimate that has a bootstrap interval is drawn with
  it. A bare bar at 0.42 over 20 queries claims a precision the data does not have.
* **The claim travels with the measurement.** The ablation figure plots each published
  delta beside the measured one; a row that could not be measured is annotated in place
  rather than dropped, so the figure cannot quietly retire an unverified number.
* **Colour carries identity only**, in a fixed order, and every bar is directly labelled
  so nothing depends on hue - which also satisfies the contrast relief rule for the
  third slot.

The palette is the validated default categorical set (slots 1-3, light surface): worst
all-pairs CVD delta-E 9.2, normal-vision 24.0. Slot 3 sits below 3:1 against the surface,
so marks in it always carry a visible value label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from docureason.tripath.utils import get_logger

logger = get_logger(__name__)

# --- palette (light surface) -------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
#: Categorical slots in fixed order. Never cycled: a fourth series folds into "other"
#: or becomes its own figure.
SERIES: Tuple[str, str, str] = ("#2a78d6", "#eb6834", "#1baf7a")

#: Quality metrics live on [0, 1] and may share an axis. Latency does not, and gets its
#: own figures.
QUALITY_METRICS: Tuple[Tuple[str, str], ...] = (
    ("average_recall_at_5", "recall@5"),
    ("average_precision_at_5", "precision@5"),
    ("average_hit_rate_at_5", "hit rate@5"),
    ("average_ndcg_at_5", "nDCG@5"),
    ("mrr", "MRR"),
    ("map_at_10", "MAP@10"),
    ("attribution_precision", "attribution precision"),
    ("hallucination_rate", "hallucination rate"),
    ("token_f1", "token F1"),
    ("exact_match", "exact match"),
)

#: Summary key -> the per-case metric its bootstrap interval is stored under.
CI_ALIASES: Dict[str, str] = {
    "average_recall_at_5": "recall_at_5",
    "average_precision_at_5": "precision_at_5",
    "average_hit_rate_at_5": "hit_rate_at_5",
    "average_ndcg_at_5": "ndcg_at_5",
    "mean_latency_ms": "latency_ms",
}

DEFAULT_FORMATS: Tuple[str, ...] = ("png", "pdf")


def _pyplot():
    """Import matplotlib lazily, with a message that says what to install.

    Plotting is not a runtime dependency of the framework - a serving container has no
    reason to carry it - so it lives in the ``figures`` extra and is imported here.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise ImportError(
            "figures require matplotlib: pip install 'docureason-framework[figures]'"
        ) from exc
    matplotlib.use("Agg")  # headless: these are files, never a window
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.dpi": 110,
        }
    )
    return plt


def _save(fig, out_dir: Path, stem: str, formats: Sequence[str]) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for extension in formats:
        path = out_dir / f"{stem}.{extension}"
        # PNG for the notebook, PDF for a paper: same figure, no re-styling.
        fig.savefig(path, dpi=200 if extension == "png" else None, bbox_inches="tight")
        written.append(path)
    fig.clf()
    _pyplot().close(fig)
    return written


def _number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _grid_x(ax) -> None:
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)


def _caption(fig, text: str) -> None:
    """One line under the figure saying what the reader is looking at and over what n."""
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=7.5, color=INK_MUTED, wrap=True)


def _top_down(ax, count: int) -> None:
    """Row 0 at the top. Idempotent, because a figure may draw two series on one axis."""
    ax.set_ylim(count - 0.6, -0.6)


def _forest(
    ax,
    labels: Sequence[str],
    points: Sequence[Optional[float]],
    lows: Sequence[Optional[float]],
    highs: Sequence[Optional[float]],
    color: str = SERIES[0],
    label: Optional[str] = None,
    offset: float = 0.0,
    marker: str = "o",
    filled: Optional[Sequence[bool]] = None,
) -> None:
    """Dot-and-interval: the point estimate with the interval it actually carries."""
    positions = [index + offset for index in range(len(labels))]
    labelled = False
    for index, position in enumerate(positions):
        point, low, high = points[index], lows[index], highs[index]
        if point is None:
            continue
        if low is not None and high is not None:
            ax.plot([low, high], [position, position], color=color, linewidth=2.0, solid_capstyle="round", zorder=2)
        is_filled = True if filled is None else bool(filled[index])
        ax.plot(
            [point],
            [position],
            marker=marker,
            markersize=7.5,
            color=color,
            markerfacecolor=color if is_filled else SURFACE,
            markeredgecolor=color,
            markeredgewidth=1.6,
            linestyle="none",
            zorder=3,
            # On the first mark actually drawn, not on index 0: the first row may be the
            # one with nothing to plot, and the series would then vanish from the legend.
            label=None if labelled or label is None else label,
        )
        labelled = True
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)


def figure_headline_metrics(suite: Dict[str, Any], out_dir: Path, formats: Sequence[str]) -> List[Path]:
    """Every [0, 1] quality metric as a point estimate with its 95% bootstrap interval."""
    plt = _pyplot()
    summary = suite.get("summary_metrics") or {}
    intervals = suite.get("confidence_intervals") or {}

    labels, points, lows, highs = [], [], [], []
    for key, label in QUALITY_METRICS:
        value = _number(summary.get(key))
        if value is None:
            continue
        interval = intervals.get(CI_ALIASES.get(key, key)) or {}
        labels.append(label)
        points.append(value)
        lows.append(_number(interval.get("ci_low")))
        highs.append(_number(interval.get("ci_high")))

    if not labels:
        logger.warning("headline metrics: no quality metrics in the report; skipping")
        return []

    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(labels) + 1.3))
    _forest(ax, labels, points, lows, highs)
    _top_down(ax, len(labels))
    for index, value in enumerate(points):
        # Direct labels: the interval carries the uncertainty, the number carries the value,
        # and neither depends on reading a position off an axis.
        ax.annotate(
            f"{value:.3f}",
            (value, index),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=INK_SECONDARY,
        )

    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("score (0-1)")
    ax.set_title("Retrieval, attribution and answer quality, with 95% bootstrap intervals")
    _grid_x(ax)
    scored = suite.get("queries_scored")
    iterations = suite.get("bootstrap_iterations")
    _caption(
        fig,
        f"n = {scored} gold queries; percentile bootstrap, {iterations} resamples. A metric with no "
        "interval was not in the harness's interval set. hallucination_rate is inverted: lower is better.",
    )
    return _save(fig, out_dir, "fig1_headline_metrics", formats)


def figure_per_stratum(suite: Dict[str, Any], out_dir: Path, formats: Sequence[str]) -> List[Path]:
    """Three metrics per stratum. The place a flat corpus-wide mean gives itself away."""
    plt = _pyplot()
    per_stratum = suite.get("per_stratum") or {}
    if not per_stratum:
        logger.warning("per-stratum figure: report has no strata; skipping")
        return []

    metrics = (("average_recall_at_5", "recall@5"), ("average_ndcg_at_5", "nDCG@5"), ("token_f1", "token F1"))
    strata = sorted(per_stratum)
    height = 0.24

    fig, ax = plt.subplots(figsize=(6.8, 1.15 * len(strata) + 1.5))
    for series_index, (key, label) in enumerate(metrics):
        offsets = [index + (series_index - 1) * (height + 0.03) for index in range(len(strata))]
        # A metric the stratum never had (prose queries carry no ground-truth answer, so
        # token F1 does not exist for them) is annotated "n/a", never drawn as a zero bar:
        # "measured zero" and "not measured" are different findings.
        raw = [_number((per_stratum[stratum].get("summary_metrics") or {}).get(key)) for stratum in strata]
        ax.barh(
            offsets,
            [value if value is not None else 0.0 for value in raw],
            height=height,
            color=SERIES[series_index],
            label=label,
            # A 2px surface gap between adjacent fills, so bars read as separate marks.
            edgecolor=SURFACE,
            linewidth=1.4,
            zorder=2,
        )
        for offset, value in zip(offsets, raw):
            ax.annotate(
                "n/a" if value is None else f"{value:.3f}",
                (value or 0.0, offset),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.2,
                color=INK_MUTED if value is None else INK_SECONDARY,
            )

    ax.set_yticks(range(len(strata)))
    ax.set_yticklabels([f"{stratum}\n(n={per_stratum[stratum].get('n', '?')})" for stratum in strata])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("score (0-1)")
    ax.set_title("Quality by query stratum", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncols=3, borderaxespad=0.0)
    _grid_x(ax)
    _caption(
        fig,
        "Every bar is labelled, so identity never rests on colour alone. A stratum whose "
        "token F1 reads 0.000 had answers scored against a ground truth it did not match, not a missing metric.",
    )
    return _save(fig, out_dir, "fig2_per_stratum", formats)


def _mean_residual_ms(suite: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Mean signed residual and mean total, over the cases that carried a breakdown."""
    residuals, totals = [], []
    for record in suite.get("per_query") or []:
        breakdown = record.get("latency_breakdown") or {}
        if breakdown.get("cached") or not breakdown.get("stages_ms"):
            continue
        residual, total = _number(breakdown.get("residual_ms")), _number(breakdown.get("total_ms"))
        if residual is not None:
            residuals.append(residual)
        if total is not None:
            totals.append(total)
    mean_residual = sum(residuals) / len(residuals) if residuals else None
    mean_total = sum(totals) / len(totals) if totals else None
    return mean_residual, mean_total


def figure_stage_latency(suite: Dict[str, Any], out_dir: Path, formats: Sequence[str]) -> List[Path]:
    """Where the time goes, per stage: mean bar with a p95 tick on the same axis."""
    plt = _pyplot()
    stages = suite.get("stage_latency_ms") or {}
    rows = [
        (name, _number(block.get("mean_ms")), _number(block.get("p95_ms")))
        for name, block in stages.items()
        if isinstance(block, dict) and _number(block.get("mean_ms")) is not None
    ]
    if not rows:
        logger.warning("stage latency figure: no per-stage breakdown in the report; skipping")
        return []

    rows.sort(key=lambda row: row[1], reverse=True)
    names = [row[0] for row in rows]
    means = [row[1] for row in rows]
    p95s = [row[2] for row in rows]
    total_mean = sum(means)

    fig, ax = plt.subplots(figsize=(6.8, 0.36 * len(names) + 1.5))
    ax.barh(range(len(names)), means, height=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=1.2, zorder=2)
    for index, (mean_ms, p95_ms) in enumerate(zip(means, p95s)):
        share = (mean_ms / total_mean * 100.0) if total_mean else 0.0
        ax.annotate(
            f"{mean_ms:,.0f} ms  ({share:.0f}%)",
            (mean_ms, index),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK_SECONDARY,
        )
        if p95_ms is not None:
            # p95 as a tick on the same scale - a second statistic of the same measure,
            # never a second axis.
            ax.plot([p95_ms, p95_ms], [index - 0.28, index + 0.28], color=INK, linewidth=1.4, zorder=3)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("milliseconds (bar = mean, vertical tick = p95)")
    ax.set_title("Per-stage latency decomposition")
    ax.set_xlim(0, max(max(means), max([value for value in p95s if value is not None], default=0)) * 1.28)
    _grid_x(ax)

    residual, total = _mean_residual_ms(suite)
    residual_note = (
        "residual not recorded"
        if residual is None
        else f"mean residual {residual:+,.0f} ms of {total:,.0f} ms total"
        + (" - negative means the parallel paths overlapped" if residual < 0 else "")
    )
    _caption(
        fig,
        f"{stages.get('cases_with_breakdown', '?')} uncached queries. Stages sum to {total_mean:,.0f} ms; "
        f"{residual_note}. Cached responses are excluded: their breakdown describes a cache lookup.",
    )
    return _save(fig, out_dir, "fig3_stage_latency", formats)


def figure_latency_distribution(suite: Dict[str, Any], out_dir: Path, formats: Sequence[str]) -> List[Path]:
    """The whole latency distribution, because a mean hides the tail that hurts."""
    plt = _pyplot()
    latencies = sorted(
        value for value in (_number(record.get("latency_ms")) for record in suite.get("per_query") or []) if value
    )
    if len(latencies) < 3:
        logger.warning("latency distribution: fewer than 3 per-query latencies; skipping")
        return []

    cumulative = [(index + 1) / len(latencies) for index in range(len(latencies))]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.step(latencies, cumulative, where="post", color=SERIES[0], linewidth=2.0, zorder=3)

    summary = suite.get("summary_metrics") or {}
    for key, name in (("latency_p50_ms", "p50"), ("latency_p90_ms", "p90"), ("latency_p99_ms", "p99")):
        value = _number(summary.get(key))
        if value is None:
            continue
        ax.axvline(value, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(
            f"{name} {value:,.0f} ms",
            (value, 0.02),
            xytext=(3, 0),
            textcoords="offset points",
            rotation=90,
            va="bottom",
            fontsize=7.2,
            color=INK_SECONDARY,
        )

    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, max(latencies) * 1.05)
    ax.set_xlabel("end-to-end query latency (ms)")
    ax.set_ylabel("fraction of queries at or below")
    ax.set_title("Latency distribution (empirical CDF)")
    ax.grid(axis="both")
    ax.set_axisbelow(True)
    _caption(fig, f"n = {len(latencies)} queries, measured around service.query() including generation and attribution.")
    return _save(fig, out_dir, "fig4_latency_distribution", formats)


def figure_ablation_effects(report: Dict[str, Any], out_dir: Path, formats: Sequence[str]) -> List[Path]:
    """Measured effect beside the published claim, one row per ablation.

    The point of the figure: a row whose claim the measurement does not support should be
    visibly unsupported, and a row that could not be measured should be visibly unmeasured
    rather than absent.
    """
    plt = _pyplot()
    rows = report.get("ablations") or []
    if not rows:
        logger.warning("ablation figure: report has no rows; skipping")
        return []

    labels, measured, lows, highs, claimed, notes, filled = [], [], [], [], [], [], []
    for row in rows:
        primary = (row.get("comparisons") or {}).get(row.get("primary_metric")) or {}
        label = str(row.get("label", row.get("key", "?"))).replace("Remove ", "")
        labels.append(f"{label}\n({row.get('primary_metric')})")
        claimed.append(_number(row.get("documented_claim_pct")))

        if not row.get("executed"):
            measured.append(None)
            lows.append(None)
            highs.append(None)
            notes.append("not executed")
            filled.append(False)
            continue
        if primary.get("comparable") is False:
            measured.append(None)
            lows.append(None)
            highs.append(None)
            notes.append("not comparable across corpora")
            filled.append(False)
            continue

        point = _number(row.get("measured_primary_delta_pct"))
        reference = _number(primary.get("mean_full"))
        measured.append(point)
        if point is not None and reference not in (None, 0):
            # The CI is in absolute units; the same denominator that produced the point
            # estimate converts it, so the interval stays the interval of that estimate.
            lows.append((_number(primary.get("delta_ci_low")) or 0.0) / reference * 100.0)
            highs.append((_number(primary.get("delta_ci_high")) or 0.0) / reference * 100.0)
        else:
            lows.append(None)
            highs.append(None)
        p_value = _number(primary.get("p_value"))
        notes.append(
            f"{row.get('claim_supported', 'n/a')}" + ("" if p_value is None else f", p={p_value:.3f}")
            if point is not None
            else "no denominator"
        )
        filled.append(bool(primary.get("significant")))

    fig, ax = plt.subplots(figsize=(7.6, 0.78 * len(labels) + 1.7))
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
    _forest(ax, labels, measured, lows, highs, color=SERIES[0], label="measured (95% CI)", filled=filled)
    _forest(
        ax,
        labels,
        claimed,
        [None] * len(labels),
        [None] * len(labels),
        color=SERIES[1],
        label="published claim",
        marker="D",
        filled=[False] * len(labels),
    )
    _top_down(ax, len(labels))

    span = max([abs(value) for value in measured + claimed if value is not None] + [10.0])
    ax.set_xlim(-span * 1.35, span * 1.35)
    for index, note in enumerate(notes):
        ax.annotate(
            note,
            (ax.get_xlim()[1], index),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7.2,
            color=INK_MUTED,
        )

    ax.set_xlabel("change in the row's primary metric when the component is removed (%)")
    # Title padded so the legend can sit between it and the plot: "lower left" inside the
    # axes landed on the bottom row's marks.
    ax.set_title("Ablation effects: measured against published claim", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncols=2, borderaxespad=0.0)
    _grid_x(ax)
    subset = report.get("query_subset") or {}
    _caption(
        fig,
        f"n = {report.get('queries')} queries from {subset.get('source')}; "
        f"{report.get('executed')} of {len(rows)} rows executed; full matrix: {subset.get('is_full_matrix')}. "
        "A filled marker is significant at alpha=0.05 by paired sign-flip test; hollow is not. "
        "Negative means removing the component made the metric worse, which is what a claim of "
        "contribution predicts.",
    )
    return _save(fig, out_dir, "fig5_ablation_effects", formats)


def figure_baseline_comparison(report: Dict[str, Any], out_dir: Path, formats: Sequence[str]) -> List[Path]:
    """Full system minus naive RAG, per metric, paired on the same queries."""
    plt = _pyplot()
    baseline = report.get("baseline") or {}
    comparisons = baseline.get("full_vs_baseline") or {}
    if not comparisons:
        logger.warning("baseline figure: no baseline comparison in the report; skipping")
        return []

    labels, points, lows, highs, filled, notes = [], [], [], [], [], []
    for key, label in QUALITY_METRICS + (("duplicate_rank_ratio", "duplicate rank ratio"),):
        metric = CI_ALIASES.get(key, key)
        comparison = comparisons.get(metric)
        if not comparison:
            continue
        delta = _number(comparison.get("delta"))
        if delta is None:
            continue
        labels.append(label)
        points.append(delta)
        lows.append(_number(comparison.get("delta_ci_low")))
        highs.append(_number(comparison.get("delta_ci_high")))
        filled.append(bool(comparison.get("significant")))
        p_value = _number(comparison.get("p_value"))
        notes.append("" if p_value is None else f"p={p_value:.3f}")

    if not labels:
        return []

    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(labels) + 1.7))
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
    _forest(ax, labels, points, lows, highs, color=SERIES[0], filled=filled)
    _top_down(ax, len(labels))
    for index, (delta, note) in enumerate(zip(points, notes)):
        ax.annotate(
            f"{delta:+.3f}  {note}",
            (delta, index),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
            color=INK_SECONDARY,
        )

    span = max(abs(value) for value in points + [value for value in lows + highs if value is not None]) or 0.1
    ax.set_xlim(-span * 1.4, span * 1.4)
    ax.set_xlabel(f"{baseline.get('system', 'baseline')} subtracted from the full system (absolute)")
    ax.set_title("Full tri-path system vs the naive-RAG baseline")
    _grid_x(ax)
    _caption(
        fig,
        "Paired on the same queries and scored through the same harness, with the same generator and the "
        "same embedding model, so the difference is the retrieval architecture. Filled markers are "
        "significant at alpha=0.05; hollow are not. Latency is excluded here - it is not on this scale - "
        "and appears in the latency figures.",
    )
    return _save(fig, out_dir, "fig6_baseline_comparison", formats)


def figure_degradation(suite: Dict[str, Any], out_dir: Path, formats: Sequence[str]) -> List[Path]:
    """Which fallback boundaries fired, and on how many queries.

    Worth a figure because it is the difference between a number produced by the
    architecture and the same number produced by its fallbacks.
    """
    plt = _pyplot()
    counts = suite.get("degradation_counts") or {}
    if not counts:
        logger.info("degradation figure: no degradation reported; skipping (that is the good case)")
        return []

    rows = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    labels = [name for name, _ in rows]
    values = [value for _, value in rows]
    scored = suite.get("queries_scored") or max(values)

    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(labels) + 1.5))
    ax.barh(range(len(labels)), values, height=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=1.2, zorder=2)
    for index, value in enumerate(values):
        ax.annotate(
            f"{value} of {scored}  ({value / scored * 100:.0f}%)",
            (value, index),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK_SECONDARY,
        )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, scored * 1.35)
    ax.set_xlabel("queries affected")
    ax.set_title("Degradation: which fallbacks fired during the run")
    _grid_x(ax)
    _caption(
        fig,
        "A label reaching 100% means that boundary fired on every query, so every metric above "
        "describes the fallback rather than the component it replaced.",
    )
    return _save(fig, out_dir, "fig7_degradation", formats)


def _sign_test_floor(n: int) -> Optional[float]:
    """Smallest two-sided p an exact paired sign-flip test can return over *n* pairs.

    Worth stating: at n = 3 the floor is 0.25, so every row in such a run reads
    "not significant" no matter how large the true effect. That is a property of the test,
    not a finding about the system, and it is the single easiest number to misread.
    """
    return 2.0 ** (1 - n) if 1 <= n <= 30 else None


def _insight_lines(suite: Optional[Dict[str, Any]], ablation: Optional[Dict[str, Any]]) -> List[str]:
    lines: List[str] = ["# Insights", ""]

    if suite:
        summary = suite.get("summary_metrics") or {}
        intervals = suite.get("confidence_intervals") or {}
        scored = suite.get("queries_scored")
        lines += [f"## What the system scores, over {scored} gold queries", ""]

        for key, label in (("average_recall_at_5", "recall@5"), ("average_ndcg_at_5", "nDCG@5")):
            value = _number(summary.get(key))
            interval = intervals.get(CI_ALIASES.get(key, key)) or {}
            low, high = _number(interval.get("ci_low")), _number(interval.get("ci_high"))
            if value is None:
                continue
            if low is not None and high is not None:
                width = high - low
                precision = (
                    " The interval is wider than the estimate, so this run bounds the metric loosely rather "
                    "than measuring it." if value and width > value else ""
                )
                lines.append(f"- **{label} = {value:.3f}**, 95% CI [{low:.3f}, {high:.3f}].{precision}")
            else:
                lines.append(f"- **{label} = {value:.3f}** (no interval recorded).")

        token_f1, exact = _number(summary.get("token_f1")), _number(summary.get("exact_match"))
        if token_f1 is not None:
            answer_note = (
                " Both are zero: the answers share no tokens with the gold strings, which for derived "
                "table-cell labels usually means the generator returned the row rather than the cell."
                if not token_f1 and not exact
                else ""
            )
            exact_text = "not measured" if exact is None else f"{exact:.3f}"
            lines.append(f"- Answer quality: token F1 {token_f1:.3f}, exact match {exact_text}.{answer_note}")

        attribution, hallucination = _number(summary.get("attribution_precision")), _number(summary.get("hallucination_rate"))
        if attribution is not None and hallucination is not None:
            lines.append(
                f"- Attribution precision {attribution:.3f}, hallucination rate {hallucination:.3f}. Read these as "
                "claim-support coverage, not entailment: the attributor is lexical plus numeric consistency."
            )

        per_stratum = suite.get("per_stratum") or {}
        scores = {
            stratum: _number((block.get("summary_metrics") or {}).get("average_recall_at_5"))
            for stratum, block in per_stratum.items()
        }
        scores = {stratum: value for stratum, value in scores.items() if value is not None}
        if len(scores) >= 2:
            best = max(scores, key=lambda name: scores[name])
            worst = min(scores, key=lambda name: scores[name])
            corpus_wide = _number(summary.get("average_recall_at_5"))
            lines += [
                "",
                "## Where it is strong and weak",
                "",
                f"- **{best}** recall@5 {scores[best]:.3f} vs **{worst}** {scores[worst]:.3f} - a spread of "
                f"{scores[best] - scores[worst]:.3f}"
                + (f" around a corpus-wide {corpus_wide:.3f}." if corpus_wide is not None else "."),
                "- A single headline number cannot show that spread, which is why the strata are reported "
                "separately rather than averaged away.",
            ]

    if suite:
        stages = suite.get("stage_latency_ms") or {}
        rows = [
            (name, _number(block.get("mean_ms")))
            for name, block in stages.items()
            if isinstance(block, dict) and _number(block.get("mean_ms")) is not None
        ]
        if rows:
            rows.sort(key=lambda row: row[1], reverse=True)
            total = sum(value for _, value in rows)
            top_two = ", ".join(f"`{name}` {value:,.0f} ms ({value / total * 100:.0f}%)" for name, value in rows[:2])
            lines += ["", "## Where the time goes", "", f"- Dominated by {top_two} of {total:,.0f} ms across stages."]
            residual, mean_total = _mean_residual_ms(suite)
            if residual is not None and mean_total:
                share = abs(residual) / mean_total * 100.0
                reading = (
                    "stages overlap, which is the parallel dispatch doing its job"
                    if residual < 0
                    else "time is spent outside the instrumented stages"
                )
                lines.append(
                    f"- Mean residual {residual:+,.0f} ms of {mean_total:,.0f} ms ({share:.0f}% unaccounted): {reading}."
                )
            p50, p90 = _number((suite.get("summary_metrics") or {}).get("latency_p50_ms")), _number(
                (suite.get("summary_metrics") or {}).get("latency_p90_ms")
            )
            if p50 and p90:
                lines.append(
                    f"- Tail: p90 is {p90 / p50:.1f}x p50 ({p90:,.0f} ms vs {p50:,.0f} ms), so a mean latency "
                    "figure understates what a slow query costs."
                )
    return lines


def _ablation_insight_lines(ablation: Dict[str, Any]) -> List[str]:
    rows = ablation.get("ablations") or []
    if not rows:
        return []

    lines = ["", "## What each component contributes", ""]
    queries = ablation.get("queries") or 0
    floor = _sign_test_floor(int(queries))
    subset = ablation.get("query_subset") or {}
    lines.append(
        f"- {ablation.get('executed')} of {len(rows)} rows executed over {queries} queries from "
        f"`{subset.get('source')}`; full matrix: **{subset.get('is_full_matrix')}**."
    )
    if floor is not None and floor > 0.05:
        lines.append(
            f"- **At n = {queries} the exact paired sign-flip test cannot return a p below {floor:.3f}.** Every "
            "row here therefore reads *not significant* regardless of effect size: that is the test's floor, "
            "not evidence the component does nothing. Raise the query count before reading these verdicts."
        )

    significant, contradicted, unmeasurable = [], [], []
    for row in rows:
        primary = (row.get("comparisons") or {}).get(row.get("primary_metric")) or {}
        name = str(row.get("label", row.get("key")))
        if not row.get("executed"):
            unmeasurable.append(f"{name} (not executed: ingestion-scoped, needs its own corpus)")
            continue
        if primary.get("comparable") is False:
            unmeasurable.append(f"{name} (not comparable: the corpus rebuild renames every gold region)")
            continue
        delta = _number(row.get("measured_primary_delta_pct"))
        if delta is None:
            unmeasurable.append(f"{name} (no denominator: the full system scores zero on this metric)")
            continue
        if primary.get("significant"):
            direction = "worse" if delta < 0 else "better"
            significant.append(f"{name}: {delta:+.1f}% on {row.get('primary_metric')} - removing it made the system {direction}")
        if row.get("claim_supported") == "contradicted":
            contradicted.append(f"{name}: measured {delta:+.1f}% against a published {row.get('documented_claim_pct'):+.1f}%")

    if significant:
        lines += ["- Significant at alpha=0.05:"] + [f"  - {item}" for item in significant]
    else:
        lines.append("- No row reached significance, so no component's contribution is established by this run.")
    if contradicted:
        lines += ["- **Claims the measurement reverses:**"] + [f"  - {item}" for item in contradicted]
    if unmeasurable:
        lines += ["- Rows this run could not turn into a number:"] + [f"  - {item}" for item in unmeasurable]

    for row in rows:
        corpora = row.get("corpora") or {}
        full, ablated = corpora.get("full") or {}, corpora.get("ablated") or {}
        if full.get("total_regions") and ablated.get("total_regions"):
            lines.append(
                f"- What the {row.get('key')} corpus rebuild actually did: regions "
                f"{full['total_regions']:,} -> {ablated['total_regions']:,}, serialized tables "
                f"{full.get('structured_tables_serialized')} -> {ablated.get('structured_tables_serialized')}. "
                "That census is the finding for this row; its recall delta is not comparable."
            )

    baseline = ablation.get("baseline") or {}
    comparisons = baseline.get("full_vs_baseline") or {}
    if comparisons:
        favours_full = [
            metric
            for metric, comparison in comparisons.items()
            if (_number(comparison.get("delta")) or 0) > 0 and comparison.get("significant")
        ]
        favours_baseline = [
            metric
            for metric, comparison in comparisons.items()
            if (_number(comparison.get("delta")) or 0) < 0 and comparison.get("significant")
        ]
        lines += ["", f"## Against {baseline.get('system', 'the baseline')}", ""]
        lines.append(
            f"- Significantly better on: {', '.join(favours_full) if favours_full else 'nothing'}. "
            f"Significantly worse on: {', '.join(favours_baseline) if favours_baseline else 'nothing'}."
        )
        latency = comparisons.get("latency_ms") or {}
        latency_delta = _number(latency.get("delta"))
        if latency_delta is not None:
            verdict = "significant" if latency.get("significant") else "not significant"
            lines.append(
                f"- Latency cost of the tri-path architecture: {latency_delta:+,.0f} ms per query ({verdict}). "
                "Whether that buys anything is exactly what the quality rows above have to answer."
            )
    return lines


def _caveat_lines(suite: Optional[Dict[str, Any]]) -> List[str]:
    lines = ["", "## What this run does not establish", ""]
    if suite:
        counts = suite.get("degradation_counts") or {}
        scored = suite.get("queries_scored") or 0
        always = [label for label, count in counts.items() if scored and count >= scored]
        if always:
            lines.append(
                f"- These boundaries fired on **every** query: {', '.join(f'`{label}`' for label in always)}. "
                "Any metric downstream of them describes the fallback, not the component it stands in for."
            )
        modes = suite.get("retrieval_modes") or {}
        scans = {label: count for label, count in modes.items() if "substring_scan" in label}
        if scans:
            lines.append(
                f"- Substring-scan fallback on: {', '.join(f'`{label}` x{count}' for label, count in scans.items())}. "
                "For those modalities the run measures a linear scan, not the index."
            )
        if modes and not scans:
            lines.append(f"- Every modality retrieved through its index: {', '.join(f'`{label}`' for label in modes)}.")
    lines += [
        "- Gold labels are **derived, not annotated**: table cells know their region by construction and prose "
        "anchors are phrases verified unique, which over-represents lookup and cannot express multi-hop reasoning.",
        "- Attribution is lexical plus numeric consistency. `entailment_threshold` stays inert until a real NLI "
        "model is loaded, so these numbers are not faithfulness measurements.",
        "- Figures are regenerated from the JSON reports beside them; every number here is traceable to a file, "
        "and none of it is carried over from a previous run.",
    ]
    return lines


def build_all(
    eval_report: Optional[Union[str, Path]] = None,
    ablation_report: Optional[Union[str, Path]] = None,
    out_dir: Union[str, Path] = "artifacts/figures",
    formats: Sequence[str] = DEFAULT_FORMATS,
) -> Dict[str, Any]:
    """Render every figure the given reports support, and write ``INSIGHTS.md`` beside them.

    A missing report is not an error - it means those figures are skipped and the digest
    says which, rather than an axis appearing with nothing on it.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    suite = _load(eval_report)
    ablation = _load(ablation_report)
    if suite is None and ablation is None:
        raise SystemExit("no reports to plot: pass --eval-report and/or --ablation-report")

    written: List[Path] = []
    skipped: List[str] = []

    def _attempt(name: str, builder, payload: Optional[Dict[str, Any]]) -> None:
        if payload is None:
            skipped.append(f"{name} (report not supplied)")
            return
        paths = builder(payload, out, formats)
        if paths:
            written.extend(paths)
        else:
            skipped.append(f"{name} (inputs absent from the report)")

    _attempt("headline metrics", figure_headline_metrics, suite)
    _attempt("per stratum", figure_per_stratum, suite)
    _attempt("stage latency", figure_stage_latency, suite)
    _attempt("latency distribution", figure_latency_distribution, suite)
    _attempt("degradation", figure_degradation, suite)
    _attempt("ablation effects", figure_ablation_effects, ablation)
    _attempt("baseline comparison", figure_baseline_comparison, ablation)

    lines = _insight_lines(suite, ablation)
    if ablation:
        lines += _ablation_insight_lines(ablation)
    lines += _caveat_lines(suite)
    if skipped:
        lines += ["", "## Figures not produced", ""] + [f"- {item}" for item in skipped]

    insights_path = out / "INSIGHTS.md"
    insights_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %d figure file(s) and %s", len(written), insights_path)
    return {"figures": [str(path) for path in written], "insights": str(insights_path), "skipped": skipped}


def _load(path: Optional[Union[str, Path]]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.exists():
        logger.warning("report not found, skipping its figures: %s", resolved)
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


__all__ = [
    "CI_ALIASES",
    "QUALITY_METRICS",
    "SERIES",
    "build_all",
    "figure_ablation_effects",
    "figure_baseline_comparison",
    "figure_degradation",
    "figure_headline_metrics",
    "figure_latency_distribution",
    "figure_per_stratum",
    "figure_stage_latency",
]

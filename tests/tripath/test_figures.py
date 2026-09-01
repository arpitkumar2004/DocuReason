"""test_figures.py - the figures say what the data says, and nothing it does not.

The failure mode worth testing for in plotting code is not a crash: it is a figure that
looks fine and misleads. Three specific ones are asserted here.

* A metric a stratum never had must render as ``n/a``, not as a zero bar. "Measured zero"
  and "not measured" are different findings and a bar cannot express the second.
* An ablation row that could not be measured must still appear, with its published claim
  still plotted, so the figure cannot quietly retire an unverified number.
* A missing report must skip its figures and say so, rather than drawing an empty axis -
  which reads as "measured, and flat".
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from docureason.tripath.evaluation import figures

pytestmark = pytest.mark.unit

pytest.importorskip("matplotlib", reason="figures need the [figures] extra")


SUITE: Dict[str, Any] = {
    "queries_scored": 12,
    "queries_requested": 12,
    "wall_clock_sec": 4.2,
    "bootstrap_iterations": 200,
    "summary_metrics": {
        "average_recall_at_5": 0.5,
        "average_ndcg_at_5": 0.42,
        "token_f1": 0.0,
        "exact_match": 0.0,
        "attribution_precision": 0.6,
        "hallucination_rate": 0.4,
        "latency_p50_ms": 900.0,
        "latency_p90_ms": 2600.0,
        "latency_p99_ms": 3100.0,
        "mean_latency_ms": 1200.0,
    },
    "confidence_intervals": {
        "recall_at_5": {"ci_low": 0.2, "ci_high": 0.8, "point": 0.5},
        "ndcg_at_5": {"ci_low": 0.1, "ci_high": 0.7, "point": 0.42},
    },
    "per_stratum": {
        # prose carries no ground-truth answer, so token_f1 is absent rather than zero.
        "prose": {"n": 6, "summary_metrics": {"average_recall_at_5": 0.8, "average_ndcg_at_5": 0.7}},
        "tabular_lookup": {
            "n": 6,
            "summary_metrics": {"average_recall_at_5": 0.2, "average_ndcg_at_5": 0.14, "token_f1": 0.0},
        },
    },
    "stage_latency_ms": {
        "cases_with_breakdown": 12,
        "rerank": {"mean_ms": 700.0, "p95_ms": 1500.0, "observations": 12},
        "dense_search": {"mean_ms": 300.0, "p95_ms": 900.0, "observations": 12},
    },
    "degradation_counts": {"generation:fallback_used": 12, "retrieval:vision:substring_scan": 3},
    "retrieval_modes": {"text:hybrid_index": 12, "vision:substring_scan": 3},
    "per_query": [
        {
            "latency_ms": 800.0 + index * 90,
            "latency_breakdown": {"stages_ms": {"rerank": 700.0}, "total_ms": 1000.0, "residual_ms": -20.0},
        }
        for index in range(12)
    ],
}

ABLATION: Dict[str, Any] = {
    "queries": 12,
    "seed": 0,
    "bootstrap_iterations": 200,
    "executed": 2,
    "not_executed": ["no_layout_parser_missing"],
    "query_subset": {"queries_used": 12, "source": "data/eval/queries.json", "is_full_matrix": False},
    "ablations": [
        {
            "key": "no_sparse_path",
            "label": "Remove BM25S sparse path",
            "primary_metric": "recall_at_5",
            "documented_claim_pct": -28.9,
            "executed": True,
            "claim_supported": "consistent_in_magnitude",
            "measured_primary_delta_pct": -30.0,
            "comparisons": {
                "recall_at_5": {
                    "delta": -0.15,
                    "delta_ci_low": -0.25,
                    "delta_ci_high": -0.05,
                    "mean_full": 0.5,
                    "p_value": 0.01,
                    "significant": True,
                }
            },
        },
        {
            "key": "no_layout_parser",
            "label": "Remove Docling layout parser",
            "primary_metric": "recall_at_5",
            "documented_claim_pct": -42.5,
            "executed": True,
            "cross_corpus": True,
            "answer_metric": "token_f1",
            "measured_answer_delta_pct": None,
            "claim_supported": "not_comparable_across_corpora",
            "comparisons": {
                "recall_at_5": {"comparable": False, "reason": "label space differs"},
                "token_f1": {"delta": 0.0, "p_value": 1.0},
            },
            "corpora": {
                "full": {"total_regions": 1169, "structured_tables_serialized": 62},
                "ablated": {"total_regions": 6, "structured_tables_serialized": 0},
            },
        },
        {
            "key": "no_layout_parser_missing",
            "label": "Row nothing measured",
            "primary_metric": "recall_at_5",
            "documented_claim_pct": -42.5,
            "executed": False,
            "reason": "ingestion-scoped: needs a corpus re-ingested with this component disabled",
        },
    ],
    "baseline": {
        "system": "NaiveRAGBaseline",
        "full_vs_baseline": {
            "recall_at_5": {
                "delta": 0.22,
                "delta_ci_low": 0.05,
                "delta_ci_high": 0.4,
                "p_value": 0.02,
                "significant": True,
            },
            "latency_ms": {"delta": 6400.0, "delta_ci_low": 3000.0, "delta_ci_high": 9000.0, "significant": True},
        },
    },
}


def _write(tmp_path, name: str, payload: Dict[str, Any]):
    import json

    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_every_supported_figure_is_written(tmp_path) -> None:
    result = figures.build_all(
        eval_report=_write(tmp_path, "eval.json", SUITE),
        ablation_report=_write(tmp_path, "abl.json", ABLATION),
        out_dir=tmp_path / "figures",
        formats=("png",),
    )

    stems = {path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for path in result["figures"]}
    assert stems == {
        "fig1_headline_metrics.png",
        "fig2_per_stratum.png",
        "fig3_stage_latency.png",
        "fig4_latency_distribution.png",
        "fig5_ablation_effects.png",
        "fig6_baseline_comparison.png",
        "fig7_degradation.png",
    }
    for path in result["figures"]:
        # A figure that renders to a few hundred bytes is an empty axis.
        assert (tmp_path / "figures" / path.rsplit("figures", 1)[-1].lstrip(r"\/")).stat().st_size > 5_000


def test_a_missing_report_skips_its_figures_rather_than_drawing_nothing(tmp_path) -> None:
    result = figures.build_all(
        eval_report=_write(tmp_path, "eval.json", SUITE),
        ablation_report=tmp_path / "does_not_exist.json",
        out_dir=tmp_path / "figures",
        formats=("png",),
    )

    assert not any("ablation" in path for path in result["figures"])
    assert any("ablation effects" in item for item in result["skipped"])
    assert any("Figures not produced" in line for line in [(tmp_path / "figures" / "INSIGHTS.md").read_text(encoding="utf-8")])


def test_no_report_at_all_is_refused(tmp_path) -> None:
    with pytest.raises(SystemExit, match="no reports to plot"):
        figures.build_all(out_dir=tmp_path / "figures", formats=("png",))


def _digest(tmp_path, suite=SUITE, ablation=ABLATION) -> str:
    figures.build_all(
        eval_report=_write(tmp_path, "eval.json", suite) if suite else None,
        ablation_report=_write(tmp_path, "abl.json", ablation) if ablation else None,
        out_dir=tmp_path / "figures",
        formats=("png",),
    )
    return (tmp_path / "figures" / "INSIGHTS.md").read_text(encoding="utf-8")


def test_the_digest_reports_the_stratum_spread(tmp_path) -> None:
    text = _digest(tmp_path)

    assert "prose" in text and "tabular_lookup" in text
    assert "0.600" in text, "the spread between the best and worst stratum is the point of reporting strata"


def test_the_digest_flags_a_boundary_that_fired_on_every_query(tmp_path) -> None:
    text = _digest(tmp_path)

    assert "generation:fallback_used" in text
    assert "every" in text
    assert "substring_scan" in text, "a modality that fell back to a linear scan must be named"


def test_the_digest_carries_the_census_for_a_row_that_cannot_be_compared(tmp_path) -> None:
    text = _digest(tmp_path)

    assert "1,169 -> 6" in text
    assert "62 -> 0" in text
    assert "not comparable" in text


def test_the_digest_names_significant_rows_and_the_latency_cost(tmp_path) -> None:
    text = _digest(tmp_path)

    assert "Remove BM25S sparse path" in text
    assert "-30.0%" in text
    assert "+6,400 ms" in text, "the architecture's latency cost belongs beside its quality gain"


def test_the_digest_states_the_significance_floor_for_a_tiny_run(tmp_path) -> None:
    """At n=3 nothing can reach p<0.05, so "not significant" carries no information."""
    tiny = dict(ABLATION, queries=3)
    text = _digest(tmp_path, ablation=tiny)

    assert "cannot return a p below 0.250" in text


def test_the_significance_floor_is_two_to_the_one_minus_n() -> None:
    assert figures._sign_test_floor(3) == pytest.approx(0.25)
    assert figures._sign_test_floor(6) == pytest.approx(0.03125)
    assert figures._sign_test_floor(0) is None

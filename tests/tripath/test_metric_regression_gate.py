"""test_metric_regression_gate.py — the gate must be able to fail.

An advisory check that reads as enforced is worse than no check (audit P3-6), so the
important assertion here is not that the gate passes on the committed baseline - it is
that a changed metric makes it fail.
"""

from __future__ import annotations

import json

import pytest

from scripts.check_metric_regression import (
    BASELINE_PATH,
    FIXTURE_PATH,
    compare,
    compute,
    flatten,
    load_fixture,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def current(tmp_path_factory) -> dict:
    return compute(load_fixture(FIXTURE_PATH), tmp_path_factory.mktemp("metrics"))


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def test_the_committed_baseline_still_holds(current, baseline) -> None:
    ok, findings = compare(current, baseline, tolerance=0.01)
    assert ok, "metric values moved; if intended, re-record with --update\n" + "\n".join(findings)


def test_the_fixture_covers_every_stratum(current) -> None:
    """A gate over one stratum would not notice a formula that only affects tables."""
    assert set(current["per_stratum"]) == {"tabular_lookup", "numeric", "prose", "figure"}
    assert current["total_queries"] == 20


def test_the_gate_fails_when_a_metric_moves(current, baseline) -> None:
    drifted = json.loads(json.dumps(baseline))
    drifted["summary_metrics"]["average_recall_at_5"] += 0.05

    ok, findings = compare(current, drifted, tolerance=0.01)
    assert ok is False
    assert any("MOVED" in finding and "average_recall_at_5" in finding for finding in findings)


def test_movement_inside_the_tolerance_is_allowed(current, baseline) -> None:
    """Floating-point reassociation must not turn into a red build."""
    nudged = json.loads(json.dumps(baseline))
    nudged["summary_metrics"]["average_recall_at_5"] += 0.005

    ok, _ = compare(current, nudged, tolerance=0.01)
    assert ok is True


def test_a_removed_metric_is_drift(current, baseline) -> None:
    reduced = json.loads(json.dumps(baseline))
    reduced["summary_metrics"].pop("mrr")

    ok, findings = compare(current, reduced, tolerance=0.01)
    assert ok is False
    assert any(finding.startswith("NEW") and "mrr" in finding for finding in findings)


def test_an_added_metric_is_drift(current, baseline) -> None:
    extended = json.loads(json.dumps(baseline))
    extended["summary_metrics"]["invented_metric"] = 0.5

    ok, findings = compare(current, extended, tolerance=0.01)
    assert ok is False
    assert any(finding.startswith("REMOVED") and "invented_metric" in finding for finding in findings)


def test_machine_dependent_metrics_are_excluded(current) -> None:
    """Gating latency would fail the build on a busy runner, not on a code change."""
    names = set(flatten(current))
    assert not [name for name in names if "latency" in name or "qps" in name or "duration" in name]


def test_the_fixture_is_frozen_output_not_a_pipeline_run() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert payload["case_count"] == len(payload["cases"])
    for case in payload["cases"]:
        assert case["results"], "each case must carry its own frozen retrieval output"
        assert case["relevant_ids"]

"""test_ablation_runner.py — the ablation matrix is executed, not asserted.

The module this replaces subtracted one pre-computed number from another and had no
caller. These tests drive the runner with a stub service whose quality depends on the
overrides it receives, so a measured delta has a known correct value.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from docureason.tripath.evaluation.ablation import (
    ABLATIONS,
    CORPUS_BOUND_METRICS,
    Ablation,
    AblationRunner,
    AblationStudy,
    corpus_signature,
    markdown_matrix,
)
from docureason.tripath.evaluation.eval_harness import EvaluationHarness

pytestmark = pytest.mark.unit

CASES: List[Dict[str, Any]] = [
    {"query": f"question {i}", "relevant_ids": [f"doc-{i}-region-0"], "stratum": "prose" if i % 2 else "numeric"}
    for i in range(12)
]


class StubService:
    """Returns the gold region at rank 1 normally, and demotes it when ablated.

    ``demote_on`` names the override key that degrades this stub, so the runner's
    measured delta for that ablation has an arithmetically known sign and size.
    """

    def __init__(self, demote_on: Optional[str] = None) -> None:
        self.demote_on = demote_on
        self.calls: List[Optional[Dict[str, Any]]] = []

    def query(self, text: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.calls.append(overrides)
        index = int(text.split()[-1])
        gold = f"doc-{index}-region-0"
        ablated = bool(overrides and self.demote_on in overrides)

        ranked = [f"doc-{index}-region-{9 - position}" for position in range(4)]
        # Gold at rank 1 when healthy, pushed off the end of the top-5 when ablated.
        results = [{"region_id": gold, "text": "gold evidence", "score": 1.0}] if not ablated else []
        results += [{"region_id": rid, "text": "filler evidence", "score": 0.5} for rid in ranked]

        return {
            "results": results,
            "retrieved_evidence": results,
            "answer": "gold evidence",
            "sql_execution": {"executed": False},
            "latency_breakdown": {"stages_ms": {"rerank": 1.0}, "total_ms": 2.0},
        }


@pytest.fixture
def harness(tmp_path) -> EvaluationHarness:
    return EvaluationHarness(output_dir=tmp_path)


def _runner(harness: EvaluationHarness, service: StubService, **kwargs: Any) -> AblationRunner:
    return AblationRunner(service=service, harness=harness, iterations=200, seed=0, **kwargs)


def test_every_query_time_ablation_is_executed(harness) -> None:
    report = _runner(harness, StubService()).run(CASES)
    executed = {row["key"] for row in report["ablations"] if row["executed"]}
    query_scoped = {ablation.key for ablation in ABLATIONS if ablation.scope == "query"}
    assert executed == query_scoped


def test_an_ingestion_scoped_ablation_is_reported_not_dropped(harness) -> None:
    """Not-measured must never look like a delta of zero."""
    report = _runner(harness, StubService()).run(CASES)
    row = next(item for item in report["ablations"] if item["key"] == "no_layout_parser")

    assert row["executed"] is False
    assert "re-ingested" in row["reason"]
    assert "comparisons" not in row
    assert report["not_executed"] == ["no_layout_parser"]


class CorpusService:
    """A service answering out of its own corpus: its own region ids, its own answers.

    Stands in for the corpus rebuilt with the layout parser off. The gold ids belong to
    the un-ablated corpus, so a different ``region_prefix`` is what a corpus rebuild
    actually does to them - it does not degrade retrieval, it renames the labels.
    """

    def __init__(self, region_prefix: str, answer: str) -> None:
        self.region_prefix = region_prefix
        self.answer = answer

    def query(self, text: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        index = int(text.split()[-1])
        results = [{"region_id": f"{self.region_prefix}-{index}-region-0", "text": "evidence", "score": 1.0}]
        return {"results": results, "retrieved_evidence": results, "answer": self.answer}


ANSWER_CASES: List[Dict[str, Any]] = [
    {
        "query": f"question {i}",
        "relevant_ids": [f"doc-{i}-region-0"],
        "ground_truth_answer": "net revenue rose",
        "stratum": "prose",
    }
    for i in range(12)
]


def _cross_corpus_report(harness: EvaluationHarness) -> Dict[str, Any]:
    runner = AblationRunner(
        service=CorpusService("doc", "net revenue rose"),
        harness=harness,
        iterations=200,
        seed=0,
        ingestion_services={"no_layout_parser": CorpusService("nolayout", "unrelated filler text")},
    )
    return runner.run(ANSWER_CASES)


def test_an_ingestion_scoped_ablation_runs_when_given_its_own_corpus(harness) -> None:
    report = _cross_corpus_report(harness)
    row = next(item for item in report["ablations"] if item["key"] == "no_layout_parser")

    assert row["executed"] is True
    assert row["cross_corpus"] is True
    assert report["not_executed"] == []


def test_gold_id_metrics_are_reported_not_comparable_across_a_corpus_rebuild(harness) -> None:
    """A -100% recall that measures the label space is worse than no number at all."""
    report = _cross_corpus_report(harness)
    row = next(item for item in report["ablations"] if item["key"] == "no_layout_parser")

    for metric in CORPUS_BOUND_METRICS:
        assert row["comparisons"][metric]["comparable"] is False, metric
        assert "region identity" in row["comparisons"][metric]["reason"]
        assert "delta" not in row["comparisons"][metric]

    # recall_at_5 is this row's documented primary metric, so the verdict has to say the
    # claim was not tested rather than inherit a verdict from a number it cannot produce.
    assert row["primary_metric"] in CORPUS_BOUND_METRICS
    assert row["claim_supported"] == "not_comparable_across_corpora"


def test_a_cross_corpus_row_still_measures_answer_quality(harness) -> None:
    report = _cross_corpus_report(harness)
    row = next(item for item in report["ablations"] if item["key"] == "no_layout_parser")

    assert row["answer_metric"] == "token_f1"
    assert row["comparisons"]["token_f1"]["mean_full"] == pytest.approx(1.0)
    assert row["comparisons"]["token_f1"]["mean_ablated"] == pytest.approx(0.0)
    assert row["measured_answer_delta_pct"] == pytest.approx(-100.0)

    table = markdown_matrix(report)
    assert "token_f1 -100.0%" in table
    assert "not_comparable_across_corpora" in table


def test_a_cross_corpus_row_falls_back_to_the_corpus_census() -> None:
    """With answer quality flat on both sides, "62 tables became 0" is the finding."""
    report = {
        "ablations": [
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
            }
        ]
    }
    table = markdown_matrix(report)

    assert "tables 62->0, regions 1169->6" in table
    assert "-42.5%" in table, "the published claim stays beside a measurement it cannot be compared with"
    assert "not_comparable_across_corpora" in table


def test_a_corpus_signature_is_absent_rather_than_invented(harness) -> None:
    """A stub has no quality_audit.json; the row must say so, not fail the sweep."""
    report = _cross_corpus_report(harness)
    row = next(item for item in report["ablations"] if item["key"] == "no_layout_parser")

    assert row["corpora"] == {"full": {}, "ablated": {}}
    assert corpus_signature(CorpusService("doc", "answer")) == {}


def test_the_runner_passes_each_ablation_its_own_overrides(harness) -> None:
    service = StubService()
    _runner(harness, service).run(CASES)
    seen = [call for call in service.calls if call]
    assert {"enable_cross_encoder": False} in seen
    assert {"enable_sparse_retrieval": False} in seen
    assert None in service.calls, "the full system must run without overrides"


def test_a_real_degradation_is_measured_as_a_negative_delta(harness) -> None:
    service = StubService(demote_on="enable_sparse_retrieval")
    report = _runner(harness, service).run(CASES)
    row = next(item for item in report["ablations"] if item["key"] == "no_sparse_path")
    recall = row["comparisons"]["recall_at_5"]

    assert recall["mean_full"] == pytest.approx(1.0)
    assert recall["mean_ablated"] == pytest.approx(0.0)
    assert recall["delta"] == pytest.approx(-1.0)
    assert recall["significant"] is True
    assert row["measured_primary_delta_pct"] == pytest.approx(-100.0)


def test_an_ablation_with_no_effect_reports_no_significant_delta(harness) -> None:
    report = _runner(harness, StubService(demote_on="something_else")).run(CASES)
    row = next(item for item in report["ablations"] if item["key"] == "no_cross_encoder")

    assert row["comparisons"]["ndcg_at_5"]["delta"] == pytest.approx(0.0)
    assert row["claim_supported"] == "not_significant"


def test_a_contradicted_claim_is_labelled_as_such(harness) -> None:
    """A published delta the measurement reverses must say so, not be quietly dropped."""
    inverted = Ablation(
        key="inverted",
        label="Component whose removal helps",
        overrides={"enable_cross_encoder": False},
        scope="query",
        primary_metric="recall_at_5",
        documented_claim_pct=-25.0,
        rationale="claims removal hurts",
    )
    # Two gold regions per query, so the full system's recall is 0.5 rather than 0 and a
    # relative delta has a denominator to be expressed against.
    cases = [{"query": f"question {i}", "relevant_ids": [f"doc-{i}-region-0", f"doc-{i}-region-1"]} for i in range(12)]

    class Improves:
        def query(self, text: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            index = int(text.split()[-1])
            found = ["doc-%d-region-0" % index]
            if overrides:  # ablated: finds both gold regions
                found.append("doc-%d-region-1" % index)
            results = [{"region_id": rid, "text": "gold evidence", "score": 1.0} for rid in found]
            return {"results": results, "retrieved_evidence": results, "answer": "gold evidence"}

    report = _runner(harness, Improves(), ablations=[inverted]).run(cases)
    row = report["ablations"][0]

    assert row["comparisons"]["recall_at_5"]["mean_full"] == pytest.approx(0.5)
    assert row["comparisons"]["recall_at_5"]["mean_ablated"] == pytest.approx(1.0)
    assert row["measured_primary_delta_pct"] == pytest.approx(100.0)
    assert row["claim_supported"] == "contradicted"


def test_a_relative_delta_against_a_zero_baseline_is_not_invented(harness) -> None:
    """0% would read as "no effect"; the truth is "no denominator"."""
    ablation = Ablation(
        key="zero_baseline",
        label="Zero-baseline case",
        overrides={"enable_cross_encoder": False},
        scope="query",
        primary_metric="recall_at_5",
        documented_claim_pct=-25.0,
        rationale="n/a",
    )

    class NeverFinds:
        def query(self, text: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            results = [{"region_id": "unrelated", "text": "filler", "score": 1.0}]
            return {"results": results, "retrieved_evidence": results, "answer": ""}

    report = _runner(harness, NeverFinds(), ablations=[ablation]).run(CASES)
    row = report["ablations"][0]

    assert row["comparisons"]["recall_at_5"]["relative_delta_pct"] is None
    assert row["claim_supported"] == "not_measurable"


def test_a_baseline_is_scored_through_the_same_harness(harness) -> None:
    class Baseline:
        def query(self, text: str) -> Dict[str, Any]:
            return {"results": [{"region_id": "wrong", "text": "unrelated", "score": 1.0}], "answer": ""}

    report = _runner(harness, StubService()).run(CASES, baseline=Baseline())
    comparison = report["baseline"]["full_vs_baseline"]["recall_at_5"]

    assert comparison["mean_full"] == pytest.approx(1.0)
    assert comparison["mean_baseline"] == pytest.approx(0.0)
    assert comparison["delta"] == pytest.approx(1.0)


def test_running_without_queries_is_refused(harness) -> None:
    with pytest.raises(ValueError, match="measures nothing"):
        _runner(harness, StubService()).run([])


def test_the_matrix_renders_measured_next_to_claimed(harness) -> None:
    report = _runner(harness, StubService(demote_on="enable_sparse_retrieval")).run(CASES)
    table = markdown_matrix(report)

    assert "| Remove BM25S sparse path |" in table
    assert "-28.9%" in table, "the published claim must stay visible beside the measurement"
    assert "not executed" in table
    # An unmeasured row keeps its claim too: dropping it would retire an unverified
    # number rather than show it standing unverified.
    assert "| Remove Docling layout parser | recall_at_5 | not executed | -42.5% |" in table


def test_the_old_entry_point_refuses_rather_than_returning_a_number() -> None:
    with pytest.raises(NotImplementedError, match="AblationRunner"):
        AblationStudy().run({}, [])

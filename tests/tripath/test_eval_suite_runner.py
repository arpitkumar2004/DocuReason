"""test_eval_suite_runner.py - the suite runner scores, censuses, and survives failures.

``run_eval_suite.py`` is the command that answers "what does DocuReason score on the gold
corpus". Two things in it are easy to get silently wrong and invisible afterwards: the
degradation census (a substring-scan run scores like an index-backed one, so the count is
the only signal) and failure handling (one exploding query must not void a two-hour
suite). Both are asserted here against a stub, so neither needs a corpus to verify.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from docureason.tripath.evaluation.eval_harness import EvaluationHarness
from scripts.run_eval_suite import run_suite

pytestmark = pytest.mark.unit


class StubService:
    """Returns the gold region at rank 1, and raises on one nominated query."""

    def __init__(self, fail_on: Optional[str] = None) -> None:
        self.fail_on = fail_on
        self.queries: List[str] = []

    def query(self, text: str) -> Dict[str, Any]:
        self.queries.append(text)
        if self.fail_on and self.fail_on == text:
            raise RuntimeError("retriever exploded")
        index = int(text.split()[-1])
        results = [{"region_id": f"doc-{index}-region-0", "text": "gold evidence", "score": 1.0}]
        return {
            "results": results,
            "retrieved_evidence": results,
            "answer": "net revenue rose",
            "sql_execution": {"executed": True},
            "latency_breakdown": {"stages_ms": {"rerank": 2.0, "generate": 1.0}, "total_ms": 4.0, "residual_ms": 1.0},
            "degradation": ["attribution:lexical", "generation:fallback_used"],
            "retrieval_stats": {
                "modality_modes": {"text": {"mode": "hybrid_index"}, "vision": {"mode": "substring_scan"}}
            },
            "cached": False,
        }


CASES: List[Dict[str, Any]] = [
    {
        "query": f"question {index}",
        "relevant_ids": [f"doc-{index}-region-0"],
        "ground_truth_answer": "net revenue rose",
        "stratum": "prose" if index % 2 else "numeric",
    }
    for index in range(6)
]


@pytest.fixture
def harness(tmp_path) -> EvaluationHarness:
    return EvaluationHarness(output_dir=tmp_path)


def test_every_case_is_scored(harness) -> None:
    report = run_suite(StubService(), harness, CASES, iterations=100)

    assert report["queries_requested"] == 6
    assert report["queries_scored"] == 6
    assert report["failures"] == []
    assert report["summary_metrics"]["average_recall_at_5"] == pytest.approx(1.0)
    assert report["summary_metrics"]["token_f1"] == pytest.approx(1.0)


def test_degradation_and_retrieval_mode_are_counted_not_just_logged(harness) -> None:
    report = run_suite(StubService(), harness, CASES, iterations=100)

    assert report["degradation_counts"] == {"attribution:lexical": 6, "generation:fallback_used": 6}
    # index_backed vs substring_scan is the difference between querying the indexes and
    # scanning past them, and a suite report that omits it scores both the same.
    assert report["retrieval_modes"] == {"text:hybrid_index": 6, "vision:substring_scan": 6}


def test_a_failing_query_is_recorded_without_voiding_the_suite(harness) -> None:
    report = run_suite(StubService(fail_on="question 3"), harness, CASES, iterations=100)

    assert report["queries_scored"] == 5
    assert len(report["failures"]) == 1
    assert report["failures"][0]["query"] == "question 3"
    assert "retriever exploded" in report["failures"][0]["error"]


def test_strata_and_stage_latency_survive_into_the_report(harness) -> None:
    report = run_suite(StubService(), harness, CASES, iterations=100)

    assert sorted(report["per_stratum"]) == ["numeric", "prose"]
    assert report["per_stratum"]["prose"]["n"] == 3
    assert report["stage_latency_ms"]["rerank"]["mean_ms"] == pytest.approx(2.0)
    assert report["stage_latency_ms"]["cases_with_breakdown"] == 6
    assert report["cache_hits"] == 0

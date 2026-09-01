"""Golden-value tests for the retrieval metric math.

Every expected number below is computed by hand in the comment above it.  The
harness contains ~360 lines of from-first-principles metric code with no reference
implementation to diff against, so a literal assertion is the only thing that can
catch a regression in it.  Recomputing the formula inside the test would assert
nothing.
"""

from __future__ import annotations

import pytest

from docureason.tripath.evaluation.eval_harness import EvaluationHarness

pytestmark = pytest.mark.unit


def _results(*region_ids: str) -> list[dict]:
    """Ranked results whose only relevant field is the canonical region id."""
    return [{"region_id": rid, "document_id": rid.split("-region-")[0], "text": f"body of {rid}"} for rid in region_ids]


RANKED = _results(*(f"d1-region-{i}" for i in range(1, 6)))


@pytest.fixture
def harness(tmp_path) -> EvaluationHarness:
    return EvaluationHarness(output_dir=tmp_path / "eval")


def test_gold_at_ranks_2_and_4(harness: EvaluationHarness) -> None:
    """Gold at ranks 2 and 4 out of 5 results.

    DCG@5  = 1/log2(3) + 1/log2(5) = 0.63092975 + 0.43067656 = 1.06160631
    IDCG@5 = 1/log2(2) + 1/log2(3) = 1.0       + 0.63092975 = 1.63092975
    nDCG@5 = 1.06160631 / 1.63092975 = 0.65092...  -> 0.6509
    MRR    = 1/2 = 0.5
    MAP@10 = (1/2 + 2/4) / |gold=2| = (0.5 + 0.5) / 2 = 0.5
    """
    metrics = harness.evaluate_single(
        query="q",
        results=RANKED,
        relevant_ids=["d1-region-2", "d1-region-4"],
    )
    assert metrics["recall_at_5"] == 1.0
    assert metrics["precision_at_5"] == 0.4
    assert metrics["hit_rate_at_5"] == 1.0
    assert metrics["ndcg_at_5"] == 0.6509
    assert metrics["mrr"] == 0.5
    assert metrics["map_at_10"] == 0.5
    assert metrics["avg_gold_rank"] == 3.0
    # variance = ((2-3)^2 + (4-3)^2) / 2 = 1.0
    assert metrics["gold_rank_variance"] == 1.0
    assert metrics["noise_ratio"] == 0.6


def test_single_gold_at_rank_1_is_perfect_ndcg(harness: EvaluationHarness) -> None:
    """One gold item retrieved first: DCG = IDCG = 1/log2(2) = 1.0, so nDCG = 1.0."""
    metrics = harness.evaluate_single(query="q", results=RANKED, relevant_ids=["d1-region-1"])
    assert metrics["ndcg_at_5"] == 1.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["precision_at_5"] == 0.2
    assert metrics["mrr"] == 1.0
    assert metrics["map_at_10"] == 1.0


def test_idcg_is_capped_at_k(harness: EvaluationHarness) -> None:
    """With |gold| = 8 and k = 5, IDCG must sum only 5 terms, not 8.

    An uncapped IDCG is the single most commonly botched detail in nDCG.

    DCG@5  = 1/log2(2) + 1/log2(3) = 1.63092975   (gold at ranks 1 and 2)
    IDCG@5 = sum_{r=1..5} 1/log2(r+1)
           = 1.0 + 0.63092975 + 0.5 + 0.43067656 + 0.38685281 = 2.94845912
    nDCG@5 = 1.63092975 / 2.94845912 = 0.55314...  -> 0.5531
    """
    gold = [f"d1-region-{i}" for i in range(1, 9)]
    metrics = harness.evaluate_single(query="q", results=_results("d1-region-1", "d1-region-2"), relevant_ids=gold)
    assert metrics["ndcg_at_5"] == 0.5531
    # 2 of 8 gold items found
    assert metrics["recall_at_5"] == 0.25


def test_no_gold_in_results_scores_zero(harness: EvaluationHarness) -> None:
    metrics = harness.evaluate_single(query="q", results=RANKED, relevant_ids=["d1-region-99"])
    assert metrics["recall_at_5"] == 0.0
    assert metrics["precision_at_5"] == 0.0
    assert metrics["ndcg_at_5"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["map_at_10"] == 0.0
    assert metrics["avg_gold_rank"] is None
    assert metrics["gold_rank_variance"] is None


def test_empty_gold_set_scores_zero_not_one(harness: EvaluationHarness) -> None:
    """No ground truth means the metric is undefined; it must not read as perfect."""
    metrics = harness.evaluate_single(query="q", results=RANKED, relevant_ids=[])
    assert metrics["recall_at_5"] == 0.0
    assert metrics["ndcg_at_5"] == 0.0
    assert metrics["map_at_10"] == 0.0


def test_duplicated_gold_region_cannot_inflate_precision(harness: EvaluationHarness) -> None:
    """The same gold region at three ranks counts once.

    Counting positions instead of distinct ids let a duplicated region report
    precision@5 = 0.6 off a single correct retrieval.
    """
    duplicated = _results("d1-region-1", "d1-region-1", "d1-region-1", "d1-region-7", "d1-region-8")
    metrics = harness.evaluate_single(query="q", results=duplicated, relevant_ids=["d1-region-1"])
    assert metrics["recall_at_5"] == 1.0
    # 1 distinct hit / 5 retrieved
    assert metrics["precision_at_5"] == 0.2
    assert metrics["ndcg_at_5"] == 1.0
    # MAP counts the first occurrence only: (1/1) / 1 = 1.0
    assert metrics["map_at_10"] == 1.0


def test_substring_ids_do_not_match(harness: EvaluationHarness) -> None:
    """``doc_1`` must not match ``doc_10``, and vice versa."""
    metrics = harness.evaluate_single(
        query="q",
        results=[{"region_id": "doc_10", "document_id": "doc_10", "text": "x"}],
        relevant_ids=["doc_1"],
    )
    assert metrics["recall_at_5"] == 0.0
    assert metrics["ndcg_at_5"] == 0.0


def test_recall_never_exceeds_one(harness: EvaluationHarness) -> None:
    metrics = harness.evaluate_single(
        query="q",
        results=_results("d1-region-1", "d1-region-2"),
        relevant_ids=["d1-region-1", "d1-region-2"],
    )
    assert metrics["recall_at_5"] == 1.0

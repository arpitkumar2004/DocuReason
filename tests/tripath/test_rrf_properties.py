"""Property tests for Reciprocal Rank Fusion and the per-modality candidate cap.

RRF's entire claim is that it combines runs by *rank*, so it is invariant to the
scale of the incoming scores.  That claim was never asserted anywhere, and two
things silently broke it: a score normalizer that rounded to 2 decimals (creating
ties that reordered items before ranks were assigned), and an identity fallback key
that embedded the item's rank.
"""

from __future__ import annotations

import pytest

from docureason.tripath.fusion.fuse import Fuser
from docureason.tripath.fusion.late_fusion import ReciprocalRankFuser, _identity_key

pytestmark = pytest.mark.unit


def _run(modality: str, scores: list[float]) -> list[dict]:
    return [
        {"modality": modality, "region_id": f"{modality}-{i}", "score": s, "text": f"text {i}"}
        for i, s in enumerate(scores)
    ]


def test_fusion_order_is_invariant_to_score_scale() -> None:
    """Multiplying every input score by 1000 must not change the fused order."""
    fuser = Fuser(rrf_k=60)
    base = _run("text", [0.9, 0.5, 0.1]) + _run("table", [0.4, 0.2])
    scaled = [dict(item, score=item["score"] * 1000.0) for item in base]

    order_base = [item["region_id"] for item in fuser.fuse([base])]
    order_scaled = [item["region_id"] for item in fuser.fuse([scaled])]
    assert order_base == order_scaled


def test_fusion_order_is_invariant_to_tiny_score_differences() -> None:
    """Scores that differ only in the 4th decimal must keep their order.

    The removed normalizer rounded to 2 decimals, which collapsed exactly this case
    into a tie and let the fused order depend on dict iteration order.
    """
    fuser = Fuser(rrf_k=60)
    run = _run("text", [0.5003, 0.5002, 0.5001])
    order = [item["region_id"] for item in fuser.fuse([run])]
    assert order == ["text-0", "text-1", "text-2"]


def test_identity_key_prefers_region_id() -> None:
    assert _identity_key({"region_id": "d1-region-3", "id": "chunk-9"}, "text") == "d1-region-3"


def test_identity_key_does_not_depend_on_rank() -> None:
    """An item with no id must hash to the same key wherever it lands in a run.

    The old fallback was ``f"{document_id}-{modality}-{rank}"``, so the same
    document at rank 1 in one run and rank 4 in another became two fused entries
    instead of one with double the consensus weight.
    """
    item = {"document_id": "d1", "text": "identical body text"}
    assert _identity_key(item, "text") == _identity_key(dict(item), "text")


def test_same_document_in_two_runs_fuses_into_one_entry() -> None:
    """Consensus across modalities must produce one entry, not two."""
    fuser = Fuser(rrf_k=60)
    shared = {"region_id": "d1-region-1", "text": "shared", "score": 0.9}
    fused = fuser.fuse([[dict(shared, modality="text")], [dict(shared, modality="table")]])
    assert len(fused) == 1
    assert sorted(fused[0]["fused_modalities"]) == ["table", "text"]


def test_rrf_score_matches_the_formula() -> None:
    """Score(d) = sum_m w_m / (k + rank_m(d)).

    One run, weight 0.45 (the default text weight), rank 1, k = 60:
        0.45 / 61 = 0.007377...  -> rrf_score rounds to 0.00738
    """
    fuser = ReciprocalRankFuser(rrf_k=60)
    fused = fuser.fuse_rrf({"text": [{"region_id": "r1", "score": 1.0, "text": "x"}]})
    assert fused[0]["rrf_score"] == 0.00738


def test_per_modality_cap_is_enforced_inside_the_fuser() -> None:
    """The cap must hold regardless of what the caller passed in.

    Without this the cross-encoder received every match, so reranking cost was
    O(all matching regions) rather than O(top_k).
    """
    fuser = Fuser(rrf_k=60, per_run_cap={"text": 2, "table": 1, "default": 2})
    fused = fuser.fuse([_run("text", [9, 8, 7, 6, 5]), _run("table", [4, 3, 2])])
    by_modality: dict[str, int] = {}
    for item in fused:
        by_modality[item["modality"]] = by_modality.get(item["modality"], 0) + 1
    assert by_modality == {"text": 2, "table": 1}


def test_cap_keeps_the_highest_scoring_candidates() -> None:
    fuser = Fuser(rrf_k=60, per_run_cap=2)
    fused = fuser.fuse([_run("text", [0.1, 0.9, 0.5])])
    assert {item["region_id"] for item in fused} == {"text-1", "text-2"}


def test_zero_cap_means_unbounded() -> None:
    fuser = Fuser(rrf_k=60, per_run_cap=0)
    assert len(fuser.fuse([_run("text", [5, 4, 3, 2, 1])])) == 5


def test_empty_input_returns_empty() -> None:
    assert Fuser(rrf_k=60).fuse([]) == []
    assert Fuser(rrf_k=60).fuse([[], []]) == []

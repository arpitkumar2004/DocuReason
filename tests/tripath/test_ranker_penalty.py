"""test_ranker_penalty.py — the short-heading penalty must never promote a heading.

Cross-encoder outputs are unbounded signed logits.  The previous implementation
multiplied by 0.5, which moves a negative logit *toward zero* and therefore raised
the rank of exactly the table-of-contents fragments the penalty exists to suppress.
These tests pin the corrected, sign-safe behaviour.
"""

from __future__ import annotations

import pytest

from docureason.tripath.retrieval.ranker import Ranker

pytestmark = pytest.mark.unit


class _FakeCrossEncoder:
    """Deterministic stand-in so the test needs no model download."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def predict(self, pairs: list[list[str]]) -> list[float]:
        assert len(pairs) == len(self._scores)
        return self._scores


def _rank_with(scores: list[float], items: list[dict], monkeypatch) -> list[dict]:
    ranker = Ranker()
    monkeypatch.setattr(ranker, "_get_cross_encoder", lambda: _FakeCrossEncoder(scores))
    return ranker.rank("quarterly revenue", items)


def test_negative_scored_heading_is_not_promoted(monkeypatch):
    """A short heading with a negative logit must stay below a longer positive body."""
    items = [
        {"text": "Balance Sheet", "modality": "text", "document_id": "d1"},  # 13 chars -> heading
        {
            "text": "Total revenue for the quarter increased to 125 million across all regions.",
            "modality": "text",
            "document_id": "d2",
        },
    ]
    # Heading scores lower than the body. Under the old multiplicative penalty
    # (-8.0 * 0.5 = -4.0) the heading would have overtaken the body at -5.0.
    ranked = _rank_with([-8.0, -5.0], items, monkeypatch)

    assert ranked[0]["document_id"] == "d2", "long body must outrank the short heading"
    assert ranked[1]["document_id"] == "d1"
    assert ranked[1]["heading_penalty_applied"] is True
    assert ranked[0]["heading_penalty_applied"] is False


def test_penalty_is_monotone_decreasing_for_both_signs(monkeypatch):
    """The penalised score must be strictly lower than the raw score, whatever the sign."""
    heading = {"text": "Notes", "modality": "text", "document_id": "d1"}

    for raw in (5.0, 0.0, -5.0):
        ranked = _rank_with([raw], [dict(heading)], monkeypatch)
        assert ranked[0]["heading_penalty_applied"] is True
        assert ranked[0]["rank_score"] < raw, f"penalty must reduce a score of {raw}"


def test_long_text_and_non_text_modalities_are_not_penalised(monkeypatch):
    items = [
        {"text": "Rev", "modality": "table", "document_id": "t1"},  # short but tabular
        {"text": "x" * 200, "modality": "text", "document_id": "d1"},  # long body
    ]
    ranked = _rank_with([1.0, 2.0], items, monkeypatch)
    assert all(item["heading_penalty_applied"] is False for item in ranked)


def test_reranker_mode_is_reported(monkeypatch):
    """Callers must be able to tell cross-encoder scores from fallback scores."""
    items = [{"text": "x" * 100, "modality": "text", "document_id": "d1"}]
    ranked = _rank_with([1.5], items, monkeypatch)
    assert ranked[0]["reranker_mode"] == "cross_encoder"

    # Force the fallback path: no model available.
    fallback_ranker = Ranker()
    monkeypatch.setattr(fallback_ranker, "_get_cross_encoder", lambda: None)
    fallback = fallback_ranker.rank("quarterly revenue", items)
    assert fallback[0]["reranker_mode"] == "token_similarity_fallback"

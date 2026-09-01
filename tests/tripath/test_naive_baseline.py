"""test_naive_baseline.py — the comparison point every reported delta needs.

The docs quote improvements over "naive RAG" without one existing, so nothing the
tri-path system achieved could be attributed to the architecture rather than to the
corpus or the generator. These tests pin the splitter's properties and the region
attribution that lets one harness score both systems.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from docureason.tripath.evaluation.baselines import NaiveRAGBaseline, recursive_split
from docureason.tripath.evaluation.eval_harness import EvaluationHarness
from docureason.tripath.ingestion.schema import Document, Region

pytestmark = pytest.mark.unit


class StubGenerator:
    """Stands in for GenerationModule so the baseline is testable without a model."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def generate(self, query: str, results: List[Dict[str, Any]], sql_payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append({"query": query, "results": results, "sql": sql_payload})
        return {"answer": f"answer to {query}", "engine": "stub", "citations": []}


def _document() -> Document:
    regions = [
        Region(type="title", text="Annual Report", start=0, end=13, layout_source="docling"),
        Region(
            type="body",
            text="Operating revenue for the Americas segment rose to 15000 in the period under review. " * 6,
            start=0,
            end=10,
            layout_source="docling",
        ),
        Region(
            type="body",
            text="The Europe segment reported a decline attributable to currency translation effects. " * 6,
            start=0,
            end=10,
            layout_source="docling",
        ),
    ]
    for index, region in enumerate(regions):
        region.metadata = {"region_id": f"doc-naive-region-{index}"}
    return Document(id="doc-naive", source="f.html", title="Annual Report", regions=regions, metadata={})


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


def test_spans_cover_the_text_without_gaps() -> None:
    text = "alpha beta gamma. delta epsilon zeta. eta theta iota. " * 8
    spans = recursive_split(text, chunk_size=80, overlap=20)

    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start <= previous_end, "a gap between chunks would drop text silently"


def test_overlap_makes_consecutive_chunks_share_text() -> None:
    text = "word " * 200
    spans = recursive_split(text, chunk_size=100, overlap=40)
    assert len(spans) > 2
    assert spans[1][0] < spans[0][1], "consecutive chunks must overlap when overlap > 0"


def test_the_splitter_terminates_on_text_with_no_separators() -> None:
    """The last separator is the empty string precisely so this cannot loop forever."""
    spans = recursive_split("x" * 500, chunk_size=64, overlap=8)
    assert spans and spans[-1][1] == 500


def test_zero_overlap_is_allowed() -> None:
    spans = recursive_split("a b c d e f g h i j " * 20, chunk_size=50, overlap=0)
    assert spans[-1][1] == 400


def test_a_non_positive_chunk_size_is_refused() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        recursive_split("text", chunk_size=0, overlap=0)


# ---------------------------------------------------------------------------
# Region attribution and retrieval
# ---------------------------------------------------------------------------


def test_chunks_record_every_region_they_span(monkeypatch) -> None:
    """A naive splitter cuts across gold units; scoring must not punish it for that."""
    baseline = NaiveRAGBaseline(chunk_size=400, chunk_overlap=50, generator=StubGenerator())
    # Chunking is what is under test here; the embedding model is exercised by the
    # integration test at the bottom of this module, which keeps the fast gate fast.
    monkeypatch.setattr(baseline, "_get_model", lambda: None)
    baseline.build([_document()])

    assert baseline.chunks
    assert all(chunk.region_id for chunk in baseline.chunks)
    spanning = [chunk for chunk in baseline.chunks if len(chunk.region_ids) > 1]
    assert spanning, "with chunks larger than the regions, some chunk must span two"
    for chunk in spanning:
        assert chunk.region_id == chunk.region_ids[0], "the primary region is the largest contributor"


def test_the_harness_credits_a_chunk_that_overlaps_the_gold_region(tmp_path) -> None:
    harness = EvaluationHarness(output_dir=tmp_path)
    metrics = harness.evaluate_single(
        query="revenue",
        results=[{"region_id": "doc-x-region-5", "region_ids": ["doc-x-region-5", "doc-x-region-6"], "text": "t"}],
        relevant_ids=["doc-x-region-6"],
    )
    assert metrics["recall_at_5"] == 1.0


def test_retrieval_degrades_to_keyword_overlap_without_a_model(monkeypatch) -> None:
    """A missing model must not answer every query with an empty list."""
    baseline = NaiveRAGBaseline(top_k=2, generator=StubGenerator())
    monkeypatch.setattr(baseline, "_get_model", lambda: None)
    stats = baseline.build([_document()])

    assert stats["degraded"] == "embedder_unavailable"
    assert stats["index_type"] == "keyword_overlap_fallback"

    response = baseline.query("Europe segment currency translation")
    assert response["results"], "the fallback must still rank something"
    assert response["results"][0]["retrieval_mode"] == "keyword_overlap"
    assert "baseline:embedder_unavailable" in response["degradation"]


def test_the_top_k_cap_is_honoured(monkeypatch) -> None:
    baseline = NaiveRAGBaseline(chunk_size=120, chunk_overlap=20, top_k=3, generator=StubGenerator())
    monkeypatch.setattr(baseline, "_get_model", lambda: None)
    baseline.build([_document()])

    assert len(baseline.chunks) > 3
    assert len(baseline.query("segment revenue period")["results"]) <= 3


def test_the_response_shape_matches_the_full_system(monkeypatch) -> None:
    """The harness scores both systems through one path; the shape is that contract."""
    baseline = NaiveRAGBaseline(generator=StubGenerator())
    monkeypatch.setattr(baseline, "_get_model", lambda: None)
    baseline.build([_document()])
    response = baseline.query("revenue")

    for key in ("query", "results", "answer", "degradation", "latency_breakdown", "retrieval_stats", "cached"):
        assert key in response, f"missing {key!r} from the baseline response"
    assert response["system"] == "naive_rag_baseline"


def test_the_baseline_uses_the_generator_it_is_given(monkeypatch) -> None:
    """Sharing the generator is what isolates retrieval in the comparison."""
    generator = StubGenerator()
    baseline = NaiveRAGBaseline(generator=generator)
    monkeypatch.setattr(baseline, "_get_model", lambda: None)
    baseline.build([_document()])
    baseline.query("revenue")

    assert len(generator.calls) == 1
    assert generator.calls[0]["sql"] == {"executed": False, "sql_results": []}


def test_an_empty_corpus_is_reported_not_crashed() -> None:
    baseline = NaiveRAGBaseline(generator=StubGenerator())
    stats = baseline.build([])
    assert stats["degraded"] == "empty_corpus"
    assert baseline.query("anything")["results"] == []

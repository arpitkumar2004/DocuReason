"""test_naive_baseline_models.py — the baseline against the real embedding stack.

Split from ``test_naive_baseline.py`` because this is the only baseline test that loads
a model. Module-level ``pytestmark`` cannot be un-set per test, and leaving it in the
``unit`` module put a 60-second model load inside the fast gate.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from docureason.tripath.evaluation.baselines import NaiveRAGBaseline
from docureason.tripath.ingestion.schema import Document, Region

pytestmark = pytest.mark.integration


class StubGenerator:
    def generate(self, query: str, results: List[Dict[str, Any]], sql_payload: Dict[str, Any]) -> Dict[str, Any]:
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


def test_the_real_embedding_path_builds_a_flat_index() -> None:
    """The one test that loads the actual model: a flat index over real vectors."""
    baseline = NaiveRAGBaseline(chunk_size=300, chunk_overlap=40, top_k=3, generator=StubGenerator())
    stats = baseline.build([_document()])

    if stats.get("degraded"):
        pytest.skip(f"embedding stack unavailable: {stats['degraded']}")

    assert stats["index_type"] == "flat_inner_product"
    assert stats["ntotal"] == len(baseline.chunks)
    assert stats["embedding_revision"] != "hub-default", "the baseline must use the pinned weights"

    response = baseline.query("Europe segment currency translation decline")
    assert response["results"][0]["retrieval_mode"] == "flat_index"

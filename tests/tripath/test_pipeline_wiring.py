"""Pipeline wiring tests: chunker, lineage, manifest join, degradation, co-location.

These assert the *connections* rather than any single component, because the audit's
dominant failure pattern was finished components that were never wired: the chunker
was tested and called by two scripts but not by the pipeline, the quality auditor was
exported and never called, and the indexes were built and never queried.

The dense embedder is stubbed so the whole run is hermetic; everything else - the
chunker, identity, the audit, the manifest and the retrievers - is the real code.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from docureason.pipeline import DocuReasonPipeline
from docureason.tripath.ingestion.docling_wrapper import DoclingWrapper
from docureason.tripath.retrieval.text_retrieval import TextRetrieval

pytestmark = pytest.mark.unit

pytest.importorskip("faiss", reason="faiss-cpu is a declared dependency")
pytest.importorskip("bm25s", reason="bm25s is a declared dependency")

_DIM = 64

DOC_TEXT = """Quarterly Financial Review

Revenue for the quarter reached twelve million dollars, an increase of fifty percent
over the prior quarter, driven primarily by enterprise licence renewals across the
North America and EMEA segments as well as a modest contribution from services.

Operating Margin Commentary

Operating margin contracted by two hundred basis points to eighteen percent as the
company absorbed hiring costs in engineering and go-to-market, partially offset by
lower cloud infrastructure spend following the migration completed in the period.

Table: Quarterly Revenue
| Quarter | Revenue | Margin |
|---------|---------|--------|
| Q1      | $8M     | 20%    |
| Q2      | $12M    | 18%    |
"""


class StubEncoder:
    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        vectors = np.zeros((len(texts), _DIM), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", str(text).lower()):
                vectors[row, hash(token) % _DIM] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1.0, norms)
        return vectors


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUREASON_DISABLE_DOCLING", "1")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "review.txt").write_text(DOC_TEXT, encoding="utf-8")
    out = tmp_path / "out"

    pipeline = DocuReasonPipeline(input_dir=corpus, output_dir=out, verbose=False)
    # Small windows so one region produces several child chunks.
    pipeline.chunker.chunk_size = 20
    pipeline.chunker.overlap = 5
    pipeline.chunker.parent_max_words = 500
    pipeline.dense_index._model = StubEncoder()

    report = pipeline.run()
    return pipeline, out, report


def test_pipeline_calls_the_chunker_and_windows_long_regions(run) -> None:
    """More chunks than regions proves windowing happened (it was 1:1 before)."""
    _, out, report = run
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    region_count = sum(len(doc["regions"]) for doc in manifest["documents"])
    assert report["chunk_count"] > region_count > 0

    chunks = manifest["chunks"]
    assert any(chunk["metadata"]["chunk_count"] > 1 for chunk in chunks)
    assert all(chunk["metadata"]["chunk_size_unit"] == "words" for chunk in chunks)
    assert any(chunk["metadata"]["ancestry_header"].startswith("Metadata Header:") for chunk in chunks)


def test_parent_text_differs_from_child_so_expansion_can_fire(run) -> None:
    """Parent expansion was inert because parent and child were the same string."""
    _, out, _ = run
    chunks = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["chunks"]
    assert any(len(chunk["metadata"]["parent_text"]) > len(chunk["text"]) for chunk in chunks)


def test_every_chunk_region_id_exists_in_the_manifest(run) -> None:
    _, out, _ = run
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    region_ids = {region["id"] for doc in manifest["documents"] for region in doc["regions"]}
    assert region_ids
    for chunk in manifest["chunks"]:
        assert chunk["region_id"] in region_ids, f"orphan chunk lineage: {chunk['id']}"


def test_every_retrieved_region_id_exists_in_the_manifest(run) -> None:
    """The join that used to fail: retrievers each invented their own id scheme."""
    pipeline, out, _ = run
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    region_ids = {region["id"] for doc in manifest["documents"] for region in doc["regions"]}

    documents = DoclingWrapper(input_dir=pipeline.input_dir, output_dir=out).load_corpus()
    retriever = TextRetrieval(dense=pipeline.dense_index, sparse=pipeline.sparse_index, config=pipeline.config)
    results = retriever.retrieve("revenue enterprise licence renewals", documents)

    assert results, "index-backed retrieval returned nothing"
    assert retriever.last_stats["mode"] == "hybrid_index"
    for item in results:
        assert item["region_id"] in region_ids, f"citation points outside the manifest: {item['region_id']}"


def test_indexes_land_beside_the_corpus_they_index(run) -> None:
    _, out, _ = run
    for name in ("corpus.json", "index.json", "manifest.json", "quality_audit.json", "faiss_text.index"):
        assert (out / name).exists(), f"{name} missing from the run directory"
    assert (out / "bm25_text").is_dir()


def test_audit_reports_observations_not_hard_coded_claims(run) -> None:
    _, out, _ = run
    audit = json.loads((out / "quality_audit.json").read_text(encoding="utf-8"))

    # The seven hard-coded capability strings are gone.
    assert "notes" not in audit
    # ArtifactQualityAuditor output is present (it was never called before).
    assert audit["artifact_quality_score"] == 1.0
    assert audit["modality_breakdown"]["text_chunks"] > 0

    caps = audit["capabilities_used"]
    assert caps["regions_by_layout_source"] == {"rule": audit["total_regions"]}
    assert caps["index_stats"]["dense"]["text"]["ntotal"] == audit["modality_breakdown"]["text_chunks"]
    assert caps["index_stats"]["dense"]["text"]["degraded"] is None
    assert caps["chunking"]["unit"] == "words"
    assert "generated_at" in audit["provenance"]


def test_degradation_is_reported_when_docling_did_not_run(run) -> None:
    """A rule-based-only parse must be visible, not indistinguishable from deep parsing."""
    _, out, report = run
    audit = json.loads((out / "quality_audit.json").read_text(encoding="utf-8"))
    assert report["degradation"] == audit["degradation"]
    assert any(label.startswith("layout:") for label in audit["degradation"])
    assert not any("faiss_unavailable" in label for label in audit["degradation"])

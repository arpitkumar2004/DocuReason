"""test_pipeline_output_dir.py — all artifacts must land in the requested run directory.

``DenseIndexBuilder`` and ``BM25SIndexBuilder`` prefer ``config.output_dir`` over their
positional ``output_dir`` argument.  ``DocuReasonPipeline`` did not sync the config to
the requested path, so JSON artifacts went to ``output_dir`` while the FAISS and BM25S
indexes went to the config default (``artifacts/``).  Two runs with different output
directories therefore overwrote each other's indexes, and any index/corpus pairing was
guaranteed to mismatch.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from docureason.pipeline import DocuReasonPipeline


@pytest.fixture()
def fast_pipeline_env(monkeypatch):
    """Skip Docling and the local SLM so this stays an artifact-layout test."""
    monkeypatch.setenv("DOCUREASON_DISABLE_DOCLING", "1")
    monkeypatch.setenv("DOCUREASON_DISABLE_SLM", "1")
    yield
    os.environ.pop("DOCUREASON_DISABLE_DOCLING", None)
    os.environ.pop("DOCUREASON_DISABLE_SLM", None)


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text(
        "Quarterly Revenue Report\n\nOverview of performance.\n\nTable: Revenue\nQuarter | Revenue\nQ1 | 100\n",
        encoding="utf-8",
    )
    return corpus


def test_indexes_are_written_beside_the_corpus(tmp_path, fast_pipeline_env):
    out_dir = tmp_path / "run-a"
    pipeline = DocuReasonPipeline(input_dir=_corpus(tmp_path), output_dir=out_dir, verbose=False)
    pipeline.run()

    # JSON artifacts
    for name in ("corpus.json", "index.json", "manifest.json", "quality_audit.json"):
        assert (out_dir / name).exists(), f"{name} must be written to the requested output_dir"

    # The index builders must have been pointed at the same directory.
    assert pipeline.dense_index.output_dir == out_dir
    assert pipeline.sparse_index.output_dir == out_dir
    assert (out_dir / "faiss_text.index").exists(), "dense index must live beside the corpus it indexes"
    assert (out_dir / "bm25_text").exists(), "sparse index must live beside the corpus it indexes"


def test_two_runs_do_not_share_index_files(tmp_path, fast_pipeline_env):
    corpus = _corpus(tmp_path)
    first, second = tmp_path / "run-1", tmp_path / "run-2"

    DocuReasonPipeline(input_dir=corpus, output_dir=first, verbose=False).run()
    DocuReasonPipeline(input_dir=corpus, output_dir=second, verbose=False).run()

    assert (first / "faiss_text.index").exists()
    assert (second / "faiss_text.index").exists()


def test_audit_reports_observed_capabilities_not_claims(tmp_path, fast_pipeline_env):
    """quality_audit.json must describe what ran, not assert a fixed capability list."""
    import json

    out_dir = tmp_path / "run-audit"
    DocuReasonPipeline(input_dir=_corpus(tmp_path), output_dir=out_dir, verbose=False).run()

    audit = json.loads((out_dir / "quality_audit.json").read_text(encoding="utf-8"))

    assert "notes" not in audit, "the hard-coded capability claim list must be gone"
    capabilities = audit["capabilities_used"]
    # Docling was disabled for this run, so every region must be attributed to the
    # rule-based fallback rather than claimed as deep layout parsing.
    assert capabilities["regions_by_layout_source"].get("docling", 0) == 0
    assert capabilities["regions_by_layout_source"]["rule"] > 0
    assert capabilities["ocr_char_threshold"] == 50
    assert audit["provenance"]["framework_version"]
    assert audit["provenance"]["generated_at"] != "."

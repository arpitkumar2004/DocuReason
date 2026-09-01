"""test_metric_id_matching.py — retrieval metrics must match gold IDs exactly.

The previous bidirectional substring test (``rel_id in doc_id or doc_id in rel_id``)
produced false positives (``"doc_1"`` matched ``"doc_10"``) and false negatives
(``"finqa_doc_01"`` matched no ``"doc-finqa_doc-<digest>"``).  The latter pinned every
retrieval metric on the extended benchmark suite to zero.
"""

from __future__ import annotations

import pytest

from docureason.tripath.evaluation.eval_harness import EvaluationHarness, resolve_gold_ids
from docureason.tripath.ingestion.schema import Document

pytestmark = pytest.mark.unit


def test_prefix_collision_is_not_a_match(tmp_path):
    harness = EvaluationHarness(output_dir=tmp_path)
    results = [{"document_id": "doc_10", "text": "unrelated content"}]

    metrics = harness.evaluate_single(query="q", results=results, relevant_ids=["doc_1"])

    assert metrics["recall_at_5"] == 0.0, "'doc_1' must not match 'doc_10'"
    assert metrics["hit_rate_at_5"] == 0.0


def test_exact_match_scores(tmp_path):
    harness = EvaluationHarness(output_dir=tmp_path)
    results = [{"document_id": "doc_1", "text": "relevant content"}]

    metrics = harness.evaluate_single(query="q", results=results, relevant_ids=["doc_1"])

    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr"] == 1.0


def test_region_id_is_matchable(tmp_path):
    """Gold truth may be expressed at region granularity, not just document."""
    harness = EvaluationHarness(output_dir=tmp_path)
    results = [{"document_id": "doc_1", "region_id": "doc_1-region-2", "text": "t"}]

    metrics = harness.evaluate_single(query="q", results=results, relevant_ids=["doc_1-region-2"])
    assert metrics["recall_at_5"] == 1.0


def test_duplicate_gold_hits_do_not_inflate_precision(tmp_path):
    """The same gold document appearing twice counts once."""
    harness = EvaluationHarness(output_dir=tmp_path)
    results = [
        {"document_id": "doc_1", "text": "a"},
        {"document_id": "doc_1", "text": "a"},
        {"document_id": "doc_9", "text": "b"},
        {"document_id": "doc_8", "text": "c"},
        {"document_id": "doc_7", "text": "d"},
    ]

    metrics = harness.evaluate_single(query="q", results=results, relevant_ids=["doc_1"])
    assert metrics["precision_at_5"] == 0.2, "one distinct relevant item out of five slots"


def test_resolve_gold_ids_maps_stems_to_canonical_ids():
    documents = [
        Document(id="doc-sample_doc_1-0123456789abcdef", source="sample_doc_1.txt", title="Report"),
        Document(id="doc-other-fedcba9876543210", source="other.pdf", title="Other"),
    ]

    resolved = resolve_gold_ids(["sample_doc_1"], documents)
    assert resolved == {"doc-sample_doc_1-0123456789abcdef"}


def test_resolve_gold_ids_keeps_unmatched_verbatim():
    resolved = resolve_gold_ids(["finqa_doc_01"], [])
    assert resolved == {"finqa_doc_01"}

"""test_identity.py — document IDs must be content-addressed.

Hashing only the path meant an edited document kept its ID, so re-ingestion never
refreshed the index and stale content was served indefinitely.
"""

from __future__ import annotations

import pytest

from docureason.tripath.ingestion.identity import IdentityManager, canonical_region_id
from docureason.tripath.ingestion.schema import Region

pytestmark = pytest.mark.unit


def test_same_content_same_id(tmp_path):
    manager = IdentityManager()
    path = tmp_path / "report.txt"
    path.write_text("Revenue 100M", encoding="utf-8")

    first = manager.build_document_id(path)
    second = manager.build_document_id(path)
    assert first == second, "re-ingesting an unchanged file must be idempotent"


def test_changed_content_changes_id(tmp_path):
    manager = IdentityManager()
    path = tmp_path / "report.txt"

    path.write_text("Revenue 100M", encoding="utf-8")
    before = manager.build_document_id(path)

    path.write_text("Revenue 125M", encoding="utf-8")
    after = manager.build_document_id(path)

    assert before != after, "editing a document must produce a new id so the index refreshes"


def test_digest_width_is_sixteen_hex_chars(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("x", encoding="utf-8")
    digest = IdentityManager().content_digest(path)
    assert len(digest) == 16
    assert all(ch in "0123456789abcdef" for ch in digest)


def test_missing_file_degrades_to_path_digest(tmp_path):
    """Ingestion must not crash on an unreadable file."""
    manager = IdentityManager()
    doc_id = manager.build_document_id(tmp_path / "does-not-exist.txt")
    assert doc_id.startswith("doc-does-not-exist-")


def test_canonical_region_id_prefers_pipeline_metadata():
    region = Region(type="body", text="body", metadata={"region_id": "doc-x-abc-region-7"})
    assert canonical_region_id("doc-x-abc", region, 3) == "doc-x-abc-region-7"


def test_canonical_region_id_falls_back_without_offset():
    """Hand-built regions get the canonical construction, with no +1 offset drift."""
    region = Region(type="body", text="body")
    assert canonical_region_id("doc-x-abc", region, 3) == "doc-x-abc-region-3"

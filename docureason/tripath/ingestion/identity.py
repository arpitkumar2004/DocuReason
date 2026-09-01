from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

# Hex characters retained from the SHA-256 digest.  16 chars = 64 bits, giving a
# ~1e-6 collision probability at 10^7 documents (birthday bound).  The previous
# 12 chars (48 bits) reached ~1% at 2.4M documents.
_DIGEST_CHARS = 16
_READ_BLOCK = 1 << 20  # 1 MiB


class IdentityManager:
    """Create stable, content-addressed document IDs and shared namespace metadata.

    The digest covers **file content and path**, so:

    * re-ingesting an unchanged file is idempotent (same ID, index stays valid);
    * re-ingesting a *modified* file yields a new ID, which is what makes stale
      index detection possible.  Hashing the path alone (the previous behaviour)
      silently kept the old ID after an edit, so the corpus never refreshed.
    """

    def content_digest(self, source_path: str | Path) -> str:
        """Return a truncated SHA-256 over file content and path."""
        path = Path(source_path)
        hasher = hashlib.sha256()
        hasher.update(str(path).encode("utf-8"))
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(_READ_BLOCK), b""):
                    hasher.update(block)
        except OSError:
            # Unreadable or missing file: degrade to a path-only digest rather
            # than failing ingestion, but the ID is then not content-addressed.
            pass
        return hasher.hexdigest()[:_DIGEST_CHARS]

    def build_document_id(self, source_path: str | Path) -> str:
        path = Path(source_path)
        return f"doc-{path.stem}-{self.content_digest(path)}"

    def build_region_id(self, document_id: str, region_index: int) -> str:
        """Canonical region ID.  Every producer of region IDs must use this."""
        return f"{document_id}-region-{region_index}"

    def build_chunk_id(self, region_id: str, chunk_index: int) -> str:
        return f"{region_id}-chunk-{chunk_index}"

    def build_metadata(self, source_path: str | Path, document_id: str) -> Dict[str, str]:
        return {
            "source_path": str(source_path),
            "document_id": document_id,
            "namespace": "tripath-v1",
        }


def canonical_region_id(document_id: str, region: Any, region_index: int) -> str:
    """Resolve the canonical region ID for a retrieved region.

    Prefers the ID the ingestion pipeline stamped into ``region.metadata`` so that
    retrieval payloads join back to ``manifest.json``.  Falls back to the canonical
    construction for hand-built ``Region`` objects (tests, ad-hoc callers).

    Retrievers previously each rebuilt this string with their own offset
    (``-region-{i+1}``, ``-table-{i}``, ``-table-{i+1}``), which produced IDs that
    did not exist in the manifest and made one physical region look like two
    fused candidates.
    """
    metadata = getattr(region, "metadata", None) or {}
    existing = metadata.get("region_id")
    if existing:
        return str(existing)
    return f"{document_id}-region-{region_index}"

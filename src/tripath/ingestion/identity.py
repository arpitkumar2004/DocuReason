from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict


class IdentityManager:
    """Create stable document IDs and shared namespace metadata for corpus artifacts."""

    def build_document_id(self, source_path: str | Path) -> str:
        path = Path(source_path)
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
        return f"doc-{path.stem}-{digest}"

    def build_region_id(self, document_id: str, region_index: int) -> str:
        return f"{document_id}-region-{region_index}"

    def build_chunk_id(self, region_id: str, chunk_index: int) -> str:
        return f"{region_id}-chunk-{chunk_index}"

    def build_metadata(self, source_path: str | Path, document_id: str) -> Dict[str, str]:
        return {
            "source_path": str(source_path),
            "document_id": document_id,
            "namespace": "tripath-v1",
        }

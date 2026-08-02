from __future__ import annotations

from typing import List

from ..ingestion.schema import Document


class VisionEncoder:
    """Placeholder vision encoder for Phase 1 scaffolding."""

    def encode(self, documents: List[Document]) -> List[dict]:
        payload: List[dict] = []
        for document in documents:
            for region in document.regions:
                if region.type == "figure":
                    payload.append({
                        "document_id": document.id,
                        "text": region.text,
                    })
        return payload

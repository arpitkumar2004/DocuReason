from __future__ import annotations

from typing import List

from ..ingestion.schema import Document


class TextEncoder:
    """A placeholder encoder that builds a lightweight text index payload."""

    def encode(self, documents: List[Document]) -> List[dict]:
        payload: List[dict] = []
        for document in documents:
            for region in document.regions:
                payload.append({
                    "document_id": document.id,
                    "region_type": region.type,
                    "text": region.text,
                })
        return payload

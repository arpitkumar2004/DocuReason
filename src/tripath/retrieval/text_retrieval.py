from __future__ import annotations

from typing import List

from ..ingestion.schema import Document


class TextRetrieval:
    """Simple keyword-based retrieval over text regions."""

    def retrieve(self, query: str, documents: List[Document]) -> List[dict]:
        query_terms = {term.lower() for term in query.split() if term}
        results: List[dict] = []
        for document in documents:
            for region in document.regions:
                if region.type in {"title", "body"}:
                    text = region.text.lower()
                    score = sum(1 for term in query_terms if term in text)
                    if score:
                        results.append({
                            "document_id": document.id,
                            "region_id": region.text[:16],
                            "score": score,
                            "text": region.text,
                            "modality": "text",
                        })
        return sorted(results, key=lambda item: item["score"], reverse=True)

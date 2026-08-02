from __future__ import annotations

from typing import List

from ..ingestion.schema import Document


class TableRetrieval:
    """Simple retrieval over table regions."""

    def retrieve(self, query: str, documents: List[Document]) -> List[dict]:
        query_terms = {term.lower() for term in query.split() if term}
        results: List[dict] = []
        for document in documents:
            for region in document.regions:
                if region.type == "table":
                    text = region.text.lower()
                    score = sum(1 for term in query_terms if term in text)
                    if score:
                        results.append({
                            "document_id": document.id,
                            "score": score,
                            "text": region.text,
                            "modality": "table",
                        })
        return sorted(results, key=lambda item: item["score"], reverse=True)

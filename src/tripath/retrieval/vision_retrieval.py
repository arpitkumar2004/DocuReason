from __future__ import annotations

from typing import List

from ..ingestion.schema import Document


class VisionRetrieval:
    """Simple retrieval over figure regions."""

    def retrieve(self, query: str, documents: List[Document]) -> List[dict]:
        query_terms = [term.lower() for term in query.split() if term]
        results: List[dict] = []
        for document in documents:
            for region in document.regions:
                if region.type == "figure":
                    text = region.text.lower()
                    matches = sum(1 for term in query_terms if term in text)
                    if matches:
                        score = matches / max(1, len(query_terms))
                        score += 0.1 if any(term in text for term in query_terms) else 0.0
                        results.append({
                            "document_id": document.id,
                            "score": round(score, 3),
                            "text": region.text,
                            "modality": "vision",
                        })
        return sorted(results, key=lambda item: item["score"], reverse=True)

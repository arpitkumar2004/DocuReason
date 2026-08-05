from __future__ import annotations

from typing import Any, Dict, List

from ..ingestion.schema import Document


class TextRetrieval:
    """Keyword-based text retrieval with Step 3 field weighting (Body 2.5x, Header 1.0x)."""

    def retrieve(self, query: str, documents: List[Document]) -> List[Dict[str, Any]]:
        query_terms = [term.lower() for term in query.split() if term]
        if not query_terms:
            return []

        results: List[Dict[str, Any]] = []
        for document in documents:
            for idx, region in enumerate(document.regions):
                if region.type in {"title", "body", "paragraph"}:
                    text = region.text.lower()
                    matches = sum(1 for term in query_terms if term in text)
                    if matches:
                        # Step 3 Field Weighting: Body 2.5x, Title/Header 1.0x
                        field_multiplier = 1.0 if region.type == "title" else 2.5
                        base_score = (matches / max(1, len(query_terms))) * field_multiplier

                        results.append({
                            "document_id": document.id,
                            "region_id": f"{document.id}-region-{idx + 1}",
                            "score": round(base_score, 3),
                            "text": region.text,
                            "parent_text": getattr(region, "text", ""),
                            "modality": "text",
                            "field_weight": field_multiplier,
                        })

        return sorted(results, key=lambda item: item["score"], reverse=True)

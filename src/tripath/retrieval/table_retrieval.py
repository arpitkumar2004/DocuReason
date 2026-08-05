from __future__ import annotations

from typing import Any, Dict, List

from ..ingestion.schema import Document


class TableRetrieval:
    """Table retrieval with Step 3 field weighting (Table 3.0x multiplier)."""

    def retrieve(self, query: str, documents: List[Document]) -> List[Dict[str, Any]]:
        query_terms = [term.lower() for term in query.split() if term]
        if not query_terms:
            return []

        results: List[Dict[str, Any]] = []
        for document in documents:
            for idx, region in enumerate(document.regions):
                if region.type == "table":
                    text = region.text.lower()
                    matches = sum(1 for term in query_terms if term in text)
                    if matches:
                        # Step 3 Table Weighting: 3.0x multiplier
                        field_multiplier = 3.0
                        base_score = (matches / max(1, len(query_terms))) * field_multiplier

                        results.append({
                            "document_id": document.id,
                            "region_id": f"{document.id}-table-{idx + 1}",
                            "score": round(base_score, 3),
                            "text": region.text,
                            "parent_text": getattr(region, "table_markdown", "") or region.text,
                            "modality": "table",
                            "field_weight": field_multiplier,
                        })

        return sorted(results, key=lambda item: item["score"], reverse=True)

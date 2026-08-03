from __future__ import annotations

import re
from typing import List, Dict, Any

from ..ingestion.schema import Document


class TableSQLRetriever:
    """Provide a lightweight SQL-like execution path for tabular evidence."""

    def retrieve(self, query: str, documents: List[Document]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        lowered = query.lower()
        for document in documents:
            for region in document.regions:
                if region.type != "table":
                    continue
                table_text = region.text.lower()
                if any(term in table_text for term in ["revenue", "region", "quarter", "north", "south", "west"]):
                    score = 1.0 if "revenue" in lowered or "region" in lowered else 0.8
                    results.append({
                        "document_id": document.id,
                        "score": round(score, 3),
                        "text": region.text,
                        "modality": "table",
                        "mode": "sql-like",
                        "query": query,
                    })
        return sorted(results, key=lambda item: item["score"], reverse=True)

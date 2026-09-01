"""table_retrieval.py — table-modality retrieval over the FAISS + BM25S indexes.

Mirrors :mod:`docureason.tripath.retrieval.text_retrieval`: the per-modality dense
and sparse indexes are queried first, the in-memory substring scan is the explicit
fallback, and the path taken is reported on every result as ``retrieval_mode``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..ingestion.identity import canonical_region_id
from ..ingestion.schema import Document
from .index_backed import IndexBackedRetriever, MergedHit

TABLE_REGION_TYPES = frozenset({"table"})

#: Step 3 field weighting: tables carry a 3.0x multiplier.
_TABLE_WEIGHT = 3.0


class TableRetrieval(IndexBackedRetriever):
    """Hybrid dense + sparse table retrieval with a 3.0x field weight."""

    modality = "table"
    region_types = TABLE_REGION_TYPES
    uses_sparse = True

    def payload_from_hit(self, hit: MergedHit) -> Dict[str, Any]:
        region = hit.region
        payload = self.base_payload(hit, self.modality, _TABLE_WEIGHT)
        payload["text"] = self.child_text(hit)
        # The serialized GFM grid is the parent context for a table hit.
        payload["parent_text"] = getattr(region, "table_markdown", "") or region.text
        return payload

    def scan(self, query: str, documents: Sequence[Document]) -> List[Dict[str, Any]]:
        query_terms = [term.lower() for term in query.split() if term]
        if not query_terms:
            return []

        results: List[Dict[str, Any]] = []
        for document in documents:
            for idx, region in enumerate(document.regions):
                if region.type not in TABLE_REGION_TYPES:
                    continue
                text = region.text.lower()
                matches = sum(1 for term in query_terms if term in text)
                if not matches:
                    continue
                results.append(
                    {
                        "document_id": document.id,
                        "region_id": canonical_region_id(document.id, region, idx),
                        "score": round((matches / max(1, len(query_terms))) * _TABLE_WEIGHT, 3),
                        "text": region.text,
                        "parent_text": getattr(region, "table_markdown", "") or region.text,
                        "modality": self.modality,
                        "field_weight": _TABLE_WEIGHT,
                        "retrieval_mode": "substring_scan",
                    }
                )

        return sorted(results, key=lambda item: item["score"], reverse=True)

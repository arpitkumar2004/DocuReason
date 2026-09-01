"""text_retrieval.py — text-modality retrieval over the FAISS + BM25S indexes.

The dense and sparse indexes built at ingestion time are now queried here.  The
in-memory substring scan is retained as an explicit fallback for the case where no
index is readable (a fresh corpus, a stub index, or ``enable_index_retrieval =
False``), and the path actually taken is reported in ``last_stats["mode"]`` and on
every result as ``retrieval_mode`` so a degraded run is never silent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..ingestion.identity import canonical_region_id
from ..ingestion.schema import Document
from .index_backed import IndexBackedRetriever, MergedHit

#: Region types that belong to the text path.
TEXT_REGION_TYPES = frozenset({"title", "body", "paragraph"})

#: Step 3 field weighting: body 2.5x, title/header 1.0x.
_TITLE_WEIGHT = 1.0
_BODY_WEIGHT = 2.5


def field_weight_for(region_type: str) -> float:
    return _TITLE_WEIGHT if region_type in ("title", "header", "section_header") else _BODY_WEIGHT


class TextRetrieval(IndexBackedRetriever):
    """Hybrid dense + sparse text retrieval with field weighting and a scan fallback."""

    modality = "text"
    region_types = TEXT_REGION_TYPES
    uses_sparse = True

    def payload_from_hit(self, hit: MergedHit) -> Dict[str, Any]:
        region = hit.region
        metadata = region.metadata or {}
        payload = self.base_payload(hit, self.modality, field_weight_for(region.type))
        payload["text"] = self.child_text(hit)
        payload["parent_text"] = metadata.get("parent_text") or region.text
        return payload

    def scan(self, query: str, documents: Sequence[Document]) -> List[Dict[str, Any]]:
        query_terms = [term.lower() for term in query.split() if term]
        if not query_terms:
            return []

        results: List[Dict[str, Any]] = []
        for document in documents:
            for idx, region in enumerate(document.regions):
                if region.type not in TEXT_REGION_TYPES:
                    continue
                text = region.text.lower()
                matches = sum(1 for term in query_terms if term in text)
                if not matches:
                    continue
                weight = field_weight_for(region.type)
                results.append(
                    {
                        "document_id": document.id,
                        "region_id": canonical_region_id(document.id, region, idx),
                        "score": round((matches / max(1, len(query_terms))) * weight, 3),
                        "text": region.text,
                        "parent_text": (region.metadata or {}).get("parent_text") or region.text,
                        "modality": self.modality,
                        "field_weight": weight,
                        "retrieval_mode": "substring_scan",
                    }
                )

        return sorted(results, key=lambda item: item["score"], reverse=True)

"""vision_retrieval.py — figure-modality retrieval over the dense caption index.

Figure regions are indexed by their caption text, so the dense index is the only
run available here (the pipeline builds BM25S for text-bearing modalities only).
The substring scan remains the fallback.

Note what this is *not*: there is no visual embedding in the query path.  Figures
are retrieved by the text of their caption, whether that caption came from BLIP-2,
CLIP, or the deterministic metadata fallback — which is why ``caption_source`` is
carried on every result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..ingestion.identity import canonical_region_id
from ..ingestion.schema import Document
from .index_backed import IndexBackedRetriever, MergedHit

VISION_REGION_TYPES = frozenset({"figure"})

_VISION_WEIGHT = 1.0


class VisionRetrieval(IndexBackedRetriever):
    """Dense caption retrieval over figure regions, with a substring-scan fallback."""

    modality = "vision"
    region_types = VISION_REGION_TYPES
    #: The pipeline builds BM25S for text-bearing modalities only.
    uses_sparse = False

    def payload_from_hit(self, hit: MergedHit) -> Dict[str, Any]:
        region = hit.region
        metadata = region.metadata or {}
        payload = self.base_payload(hit, self.modality, _VISION_WEIGHT)
        payload["text"] = self.child_text(hit)
        payload["parent_text"] = metadata.get("parent_text") or region.text
        payload["page_no"] = region.page_no
        payload["caption_source"] = metadata.get("caption_source")
        return payload

    def scan(self, query: str, documents: Sequence[Document]) -> List[Dict[str, Any]]:
        query_terms = [term.lower() for term in query.split() if term]
        if not query_terms:
            return []

        results: List[Dict[str, Any]] = []
        for document in documents:
            for idx, region in enumerate(document.regions):
                if region.type not in VISION_REGION_TYPES:
                    continue
                text = region.text.lower()
                matches = sum(1 for term in query_terms if term in text)
                if not matches:
                    continue
                metadata = region.metadata or {}
                results.append(
                    {
                        "document_id": document.id,
                        # Was missing entirely, so RRF fell back to a rank-embedding
                        # identity key and the same figure fused as several entries.
                        "region_id": canonical_region_id(document.id, region, idx),
                        "score": round(matches / max(1, len(query_terms)) + 0.1, 3),
                        "text": region.text,
                        "parent_text": metadata.get("parent_text") or region.text,
                        "modality": self.modality,
                        "field_weight": _VISION_WEIGHT,
                        "page_no": region.page_no,
                        "caption_source": metadata.get("caption_source"),
                        "retrieval_mode": "substring_scan",
                    }
                )

        return sorted(results, key=lambda item: item["score"], reverse=True)

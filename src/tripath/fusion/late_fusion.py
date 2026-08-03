from __future__ import annotations

from typing import List, Dict, Any


class LateFusionLayer:
    """A lightweight late-fusion module for combining multi-path retrieval scores."""

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.weights = weights or {"text": 0.45, "table": 0.35, "vision": 0.20}

    def fuse(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []
        fused: List[Dict[str, Any]] = []
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in results:
            doc_id = item.get("document_id", "unknown")
            grouped.setdefault(doc_id, []).append(item)

        for doc_id, items in grouped.items():
            aggregated_score = 0.0
            modalities = []
            for item in items:
                modality = item.get("modality", "text")
                weight = self.weights.get(modality, 0.25)
                aggregated_score += float(item.get("score", 0.0)) * weight
                modalities.append(modality)
            fused.append({
                "document_id": doc_id,
                "score": round(aggregated_score, 3),
                "modalities": modalities,
                "evidence": items,
            })

        return sorted(fused, key=lambda item: item["score"], reverse=True)

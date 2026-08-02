from __future__ import annotations

from typing import List


class Ranker:
    """Rank retrieval results with a simple weighted score."""

    def rank(self, results: List[dict]) -> List[dict]:
        ranked: List[dict] = []
        for item in results:
            modality_weight = {"text": 1.0, "table": 1.2, "vision": 1.1}.get(item.get("modality"), 1.0)
            ranked.append({
                **item,
                "rank_score": round(item.get("score", 0) * modality_weight, 2),
            })
        return sorted(ranked, key=lambda entry: entry["rank_score"], reverse=True)

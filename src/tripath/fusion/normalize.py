from __future__ import annotations

from typing import List


class Normalizer:
    """Normalize retrieval scores into a common scale."""

    def normalize(self, results: List[dict]) -> List[dict]:
        if not results:
            return []
        max_score = max(item["score"] for item in results)
        normalized: List[dict] = []
        for item in results:
            normalized.append({
                **item,
                "score": round(item["score"] / max_score, 2) if max_score else 0.0,
            })
        return normalized

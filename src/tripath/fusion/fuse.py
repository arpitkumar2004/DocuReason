from __future__ import annotations

from typing import List

from .normalize import Normalizer


class Fuser:
    """Combine retrieval results from multiple modalities."""

    def fuse(self, modality_results: List[List[dict]]) -> List[dict]:
        normalized = []
        for batch in modality_results:
            normalized.extend(Normalizer().normalize(batch))
        return sorted(normalized, key=lambda item: item["score"], reverse=True)

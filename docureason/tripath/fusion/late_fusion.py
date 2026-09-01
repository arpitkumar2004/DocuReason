from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from docureason.tripath.utils import get_logger, trace_execution

logger = get_logger(__name__)


def _identity_key(item: dict, modality: str) -> str:
    """Stable identity for an RRF candidate.

    The fallback used to be ``f"{document_id}-{modality}-{rank}"``, which embedded
    the item's *rank*.  The same document arriving at different ranks in different
    runs therefore became two separate fused entries, defeating the consensus
    mechanism that is RRF's entire purpose.  The fallback is now content-derived,
    so identity is a property of the item and never of where it happened to land.
    """
    for key in ("region_id", "id", "chunk_id"):
        value = item.get(key)
        if value:
            return str(value)

    document_id = item.get("document_id") or "unknown"
    text = str(item.get("text") or item.get("linearized") or "")
    digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]
    return f"{document_id}-{modality}-{digest}"


class ReciprocalRankFuser:
    """Reciprocal Rank Fusion (RRF) module combining multi-modal retrieval lists."""

    def __init__(
        self,
        rrf_k: int = 60,
        modality_weights: Dict[str, float] | None = None,
    ) -> None:
        self.rrf_k = rrf_k
        self.weights = modality_weights or {"text": 0.45, "table": 0.35, "vision": 0.20}

    @trace_execution(logger=logger)
    def fuse_rrf(
        self,
        modality_runs: Dict[str, List[Dict[str, Any]]],
        router_weights: Dict[str, float] | None = None,
    ) -> List[Dict[str, Any]]:
        """Combine per-modality retrieval lists using weighted Reciprocal Rank Fusion.

        Formula: Score(d) = sum_{m} w_m / (k + rank_m(d))
        """
        weights = router_weights or self.weights
        rrf_scores: Dict[str, float] = {}
        item_store: Dict[str, Dict[str, Any]] = {}
        modalities_seen: Dict[str, List[str]] = {}

        for modality, items in modality_runs.items():
            if not items:
                continue
            m_weight = weights.get(modality, 0.33)
            # Sort modality items by score descending
            sorted_items = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)

            for rank, item in enumerate(sorted_items, start=1):
                item_id = _identity_key(item, modality)
                item_store[item_id] = item
                modalities_seen.setdefault(item_id, []).append(modality)

                contribution = m_weight / (self.rrf_k + rank)
                rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + contribution

        fused: List[Dict[str, Any]] = []
        for item_id, rrf_score in rrf_scores.items():
            base_item = dict(item_store[item_id])
            base_item["rrf_score"] = round(rrf_score, 5)
            base_item["score"] = round(rrf_score * 100.0, 3)  # Scale for downstream normalization
            base_item["fused_modalities"] = list(set(modalities_seen.get(item_id, [])))
            fused.append(base_item)

        return sorted(fused, key=lambda x: x["score"], reverse=True)


class LateFusionLayer:
    """Late-fusion layer wrapper over Reciprocal Rank Fusion."""

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.fuser = ReciprocalRankFuser(modality_weights=weights)

    def fuse(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []
        runs: Dict[str, List[Dict[str, Any]]] = {}
        for item in results:
            mod = item.get("modality", "text")
            runs.setdefault(mod, []).append(item)
        return self.fuser.fuse_rrf(runs)

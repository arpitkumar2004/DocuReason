from __future__ import annotations

from typing import Dict, List, Any

from src.tripath.utils import get_logger, trace_execution
from .late_fusion import ReciprocalRankFuser
from .normalize import Normalizer

logger = get_logger(__name__)


class Fuser:
    """Combine retrieval results from multiple modalities using Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_fuser = ReciprocalRankFuser(rrf_k=rrf_k)
        self.normalizer = Normalizer()

    @trace_execution(logger=logger)
    def fuse(
        self,
        modality_results: List[List[Dict[str, Any]]],
        router_weights: Dict[str, float] | None = None,
    ) -> List[Dict[str, Any]]:
        if not modality_results:
            return []

        runs: Dict[str, List[Dict[str, Any]]] = {}
        for batch in modality_results:
            if not batch:
                continue
            normalized_batch = self.normalizer.normalize(batch)
            for item in normalized_batch:
                modality = item.get("modality", "text")
                runs.setdefault(modality, []).append(item)

        return self.rrf_fuser.fuse_rrf(runs, router_weights=router_weights)

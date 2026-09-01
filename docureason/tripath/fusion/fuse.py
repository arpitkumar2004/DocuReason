from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from docureason.tripath.utils import get_logger, trace_execution

from .late_fusion import ReciprocalRankFuser

logger = get_logger(__name__)


class Fuser:
    """Combine retrieval results from multiple modalities using Reciprocal Rank Fusion.

    Note on normalization
    ---------------------
    A ``Normalizer.normalize()`` call used to sit in front of the fuser.  RRF
    consumes only *rank order*, and max-scaling is monotone, so normalization
    could never change the fused result — but it rounded scores to 2 decimals,
    which **created ties** that reordered items before ranks were assigned.  The
    step is gone; :class:`~docureason.tripath.fusion.normalize.Normalizer` is kept
    for a future score-combining fuser.
    """

    def __init__(self, rrf_k: int = 60, per_run_cap: Union[int, Dict[str, int], None] = None) -> None:
        self.rrf_fuser = ReciprocalRankFuser(rrf_k=rrf_k)
        #: Hard cap re-applied to every per-modality run inside :meth:`fuse`, so the
        #: candidate budget holds regardless of what the caller passed in.  Either a
        #: single int for all modalities or a ``{modality: cap}`` mapping.
        self.per_run_cap: Union[int, Dict[str, int]] = 0 if per_run_cap is None else per_run_cap

    @staticmethod
    def _cap_for(cap: Union[int, Dict[str, int]], modality: str) -> int:
        if isinstance(cap, dict):
            return int(cap.get(modality, cap.get("default", 0)) or 0)
        return int(cap or 0)

    @trace_execution(logger=logger)
    def fuse(
        self,
        modality_results: List[List[Dict[str, Any]]],
        router_weights: Optional[Dict[str, float]] = None,
        per_run_cap: Union[int, Dict[str, int], None] = None,
    ) -> List[Dict[str, Any]]:
        if not modality_results:
            return []

        cap = self.per_run_cap if per_run_cap is None else per_run_cap

        runs: Dict[str, List[Dict[str, Any]]] = {}
        for batch in modality_results:
            if not batch:
                continue
            for item in batch:
                modality = item.get("modality", "text")
                runs.setdefault(modality, []).append(item)

        for modality, items in runs.items():
            modality_cap = self._cap_for(cap, modality)
            if modality_cap > 0 and len(items) > modality_cap:
                items.sort(key=lambda entry: float(entry.get("score", 0.0)), reverse=True)
                logger.info(
                    "Fuser: capping modality=%s run from %d to %d candidates",
                    modality,
                    len(items),
                    modality_cap,
                )
                runs[modality] = items[:modality_cap]

        return self.rrf_fuser.fuse_rrf(runs, router_weights=router_weights)

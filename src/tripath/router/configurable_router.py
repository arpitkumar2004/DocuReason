from __future__ import annotations

import math
from typing import Dict, List, Optional

from src.tripath.utils import get_logger, trace_execution

logger = get_logger(__name__)


class ConfigurableRouter:
    """Configurable multi-modal router supporting boolean flags, probabilities, and weights."""

    def __init__(
        self,
        config: Optional[Dict[str, List[str]]] = None,
        threshold: float = 0.35,
    ) -> None:
        self.config = config or {
            "text": ["revenue", "growth", "report", "overview", "company", "margin", "statement", "financial", "narrative"],
            "table": ["table", "region", "quarter", "by", "revenue", "sum", "average", "total", "rate", "percent"],
            "vision": ["chart", "figure", "image", "graph", "adoption", "bar", "pie", "diagram", "plot", "trend"],
        }
        self.threshold = threshold

    @trace_execution(logger=logger)
    def route(self, query: str) -> Dict[str, bool]:
        """Return boolean activation status for each modality."""
        probs = self.route_probabilities(query)
        return {modality: prob >= self.threshold for modality, prob in probs.items()}

    @trace_execution(logger=logger)
    def route_probabilities(self, query: str) -> Dict[str, float]:
        """Calculate soft probability scores (0.0 to 1.0) for each modality based on query intent."""
        lowered = query.lower()
        words = set(lowered.split())
        probs: Dict[str, float] = {}

        for modality, keywords in self.config.items():
            matches = sum(1 for kw in keywords if kw in lowered or kw in words)
            if matches == 0:
                # Default baseline priority if no explicit keywords match
                score = 0.5 if modality == "text" else 0.1
            else:
                # Sigmoid scaling based on keyword match density
                score = 1.0 / (1.0 + math.exp(-1.2 * matches))
            probs[modality] = round(score, 3)

        return probs

    @trace_execution(logger=logger)
    def get_route_weights(self, query: str) -> Dict[str, float]:
        """Return normalized weights across modalities summing to 1.0 for RRF fusion."""
        probs = self.route_probabilities(query)
        total = sum(probs.values())
        if total <= 0:
            return {"text": 0.5, "table": 0.3, "vision": 0.2}
        return {m: round(score / total, 3) for m, score in probs.items()}

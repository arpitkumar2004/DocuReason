from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from src.tripath.utils import get_logger, trace_execution

from .configurable_router import ConfigurableRouter

logger = get_logger(__name__)


class Router:
    """Multi-modal router supporting model artifacts and heuristic fallback."""

    def __init__(self, model_path: Optional[str | Path] = None) -> None:
        self.configurable = ConfigurableRouter()
        self.model_data: Optional[Dict] = None

        if model_path:
            p = Path(model_path)
            if p.exists():
                try:
                    self.model_data = json.loads(p.read_text(encoding="utf-8"))
                    logger.info("Loaded trained router model from %s", p)
                except Exception as exc:
                    logger.warning("Failed loading router model from %s: %s", p, exc)

    @trace_execution(logger=logger)
    def route(self, query: str) -> Dict[str, bool]:
        return self.configurable.route(query)

    @trace_execution(logger=logger)
    def route_probabilities(self, query: str) -> Dict[str, float]:
        if self.model_data and "feature_vocabulary" in self.model_data:
            vocab = self.model_data["feature_vocabulary"]
            tokens = query.lower().split()
            scores: Dict[str, float] = {}
            for modality in ["text", "table", "vision"]:
                m_vocab = vocab.get(modality, {})
                score = sum(m_vocab.get(t, 0.0) for t in tokens)
                scores[modality] = round(min(1.0, score / max(1.0, len(tokens) * 2.0)), 3)
            return scores
        return self.configurable.route_probabilities(query)

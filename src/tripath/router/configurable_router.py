from __future__ import annotations

from typing import Dict, List, Optional


class ConfigurableRouter:
    """A simple configurable router for Phase 2."""

    def __init__(self, config: Optional[Dict[str, List[str]]] = None) -> None:
        self.config = config or {
            "text": ["revenue", "growth", "report", "overview", "company"],
            "table": ["table", "region", "quarter", "by"],
            "vision": ["chart", "figure", "image", "adoption"],
        }

    def route(self, query: str) -> Dict[str, bool]:
        lowered = query.lower()
        return {
            modality: any(term in lowered for term in terms)
            for modality, terms in self.config.items()
        }

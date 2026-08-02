from __future__ import annotations

from typing import Dict


class Router:
    """A lightweight heuristic router for Phase 2."""

    def route(self, query: str) -> Dict[str, bool]:
        lowered = query.lower()
        routes = {
            "text": any(token in lowered for token in ["revenue", "growth", "report", "overview"]),
            "table": any(token in lowered for token in ["table", "region", "by", "quarter"]),
            "vision": any(token in lowered for token in ["chart", "figure", "image", "adoption"]),
        }
        return routes

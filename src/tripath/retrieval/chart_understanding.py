from __future__ import annotations

from typing import List, Dict, Any


class ChartUnderstandingModule:
    """A lightweight chart understanding wrapper that linearizes chart-like regions into structured evidence."""

    def understand(self, query: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not evidence:
            return []
        normalized: List[Dict[str, Any]] = []
        for item in evidence:
            text = item.get("text", "")
            if "chart" in text.lower() or "figure" in text.lower() or "adoption" in text.lower():
                normalized.append({
                    **item,
                    "chart_type": "bar",
                    "linearized": "chart: adoption by region",
                    "reasoning": f"Chart evidence for query: {query}",
                })
        return normalized

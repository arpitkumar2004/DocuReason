from typing import Any, Dict, List


class ChartUnderstandingModule:
    """Dynamic chart understanding engine that linearizes visual figure evidence into structured representations."""

    def __init__(self) -> None:
        #: Visual candidates enriched by the most recent understand() call.
        self.charts_enriched = 0

    def understand(self, query: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return *evidence* with every visual candidate enriched in place.

        This used to return only the visual subset as **new** dicts, which the
        caller then concatenated onto the fused list — so every chart appeared
        twice, consuming two context slots and inflating the duplicate-rank ratio.
        Enriching in place keeps one entry per candidate and makes the return value
        a drop-in replacement for the input list.
        """
        if not evidence:
            return []
        enriched: List[Dict[str, Any]] = []
        self.charts_enriched = 0

        for item in evidence:
            modality = item.get("modality", "")
            text = str(item.get("text", "") or item.get("linearized", ""))
            metadata = item.get("metadata", {}) or {}

            # Identify if candidate is visual figure / chart
            is_visual = modality == "vision" or any(
                kw in text.lower() for kw in ("chart", "figure", "graph", "diagram", "plot")
            )
            if not is_visual:
                enriched.append(item)
                continue

            # Infer chart type dynamically from metadata or text
            chart_type = metadata.get("clip_chart_type") or metadata.get("figure_type") or self._detect_chart_type(text)

            # Linearize caption text and spatial coordinates
            page_no = metadata.get("page_no") or item.get("page_no") or "N/A"
            linearized_text = f"[{chart_type.upper()} - Page {page_no}] {text.strip()}"

            enriched.append(
                {
                    **item,
                    "modality": "vision",
                    "chart_type": chart_type,
                    "linearized": linearized_text,
                    "reasoning": f"Extracted visual chart evidence matching '{query}' (Type: {chart_type}, Page: {page_no})",
                }
            )
            self.charts_enriched += 1

        return enriched

    @staticmethod
    def _detect_chart_type(text: str) -> str:
        lowered = text.lower()
        if "bar" in lowered:
            return "bar_chart"
        if "line" in lowered or "trend" in lowered:
            return "line_chart"
        if "pie" in lowered:
            return "pie_chart"
        if "scatter" in lowered:
            return "scatter_plot"
        if "diagram" in lowered:
            return "diagram"
        return "chart"

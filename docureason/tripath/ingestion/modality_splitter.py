from __future__ import annotations

from typing import Dict, List

from .schema import Document, Region


class ModalitySplitter:
    """Split a document into simple modality-oriented region groups."""

    def split(self, document: Document) -> Dict[str, List[Region]]:
        grouped: Dict[str, List[Region]] = {"text": [], "table": [], "vision": []}
        for region in document.regions:
            if region.type == "table":
                grouped["table"].append(region)
            elif region.type == "figure":
                grouped["vision"].append(region)
            else:
                grouped["text"].append(region)
        return grouped

from __future__ import annotations

import re
from typing import Any, Dict, List


class CitationChecker:
    """Parse citation tags from generated answers and verify simple support against evidence."""

    def check(self, answer: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        tags = re.findall(r"\[(T|TAB|FIG)\d+\]", answer)
        cited_evidence = []
        for tag in tags:
            matched = next((item for item in evidence if self._tag_matches(item, tag)), None)
            if matched:
                cited_evidence.append({"tag": tag, "evidence": matched})

        supported = len(cited_evidence) > 0
        return {
            "tags": tags,
            "cited_evidence": cited_evidence,
            "supported": supported,
            "attribution_precision": round(1.0 if supported else 0.0, 3),
        }

    def _tag_matches(self, evidence: Dict[str, Any], tag: str) -> bool:
        modality = evidence.get("modality", "text")
        if tag.startswith("T"):
            return modality == "text"
        if tag.startswith("TAB"):
            return modality == "table"
        if tag.startswith("FIG"):
            return modality == "vision"
        return False

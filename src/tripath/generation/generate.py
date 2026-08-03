from __future__ import annotations

from typing import List


class GenerationModule:
    """A minimal answer generation module that grounds output in retrieval evidence."""

    def generate(self, query: str, evidence: List[dict]) -> str:
        if not evidence:
            return "No supporting evidence found."
        top = evidence[0]
        return f"Using evidence from {top.get('modality', 'unknown')} modality: {top.get('text', '')[:160]}"

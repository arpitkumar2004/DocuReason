from __future__ import annotations

from typing import List, Dict, Any


class PromptBuilder:
    """Assemble a structured prompt with citation tags for multi-modal evidence."""

    def build(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        parts = [f"Question: {query}", "Use the provided evidence and cite each answer with tags such as [T1], [TAB1], or [FIG1].", "Context:"]
        for index, item in enumerate(evidence[:5], start=1):
            modality = item.get("modality", "text")
            label = self._label_for_modality(modality, index)
            text = item.get("text", "")
            parts.append(f"{label}: {text}")
        parts.append("Answer with a brief explanation and explicit citation tags.")
        return "\n".join(parts)

    def _label_for_modality(self, modality: str, index: int) -> str:
        mapping = {"text": f"[T{index}]", "table": f"[TAB{index}]", "vision": f"[FIG{index}]"}
        return mapping.get(modality, f"[CTX{index}]")

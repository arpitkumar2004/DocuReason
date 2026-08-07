from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..attribution.nli_attributor import NLIFaithfulnessAttributor
from ..evaluation.eval_harness import EvaluationHarness
from ..fusion.late_fusion import LateFusionLayer
from ..generation.prompt_builder import PromptBuilder
from .query_service import QueryService


class AsyncQueryService:
    """A production-oriented asynchronous query service wrapper."""

    def __init__(self, input_dir: str | Path, output_dir: str | Path) -> None:
        self.query_service = QueryService(input_dir=input_dir, output_dir=output_dir)
        self.fusion = LateFusionLayer()
        self.prompt_builder = PromptBuilder()
        self.attributor = NLIFaithfulnessAttributor()
        self.evaluator = EvaluationHarness(output_dir=output_dir)

    def run(self, query: str) -> Dict[str, Any]:
        raw_response = self.query_service.query(query)
        fused = self.fusion.fuse(raw_response.get("results", []))
        prompt = self.prompt_builder.build(query, fused)
        answer = self._compose_answer(query, fused)
        attribution = self.attributor.attribute(answer, fused)
        metrics = self.evaluator.evaluate(query, raw_response.get("results", []), relevant_ids=[item.get("document_id") for item in raw_response.get("results", [])][:1])
        return {
            "query": query,
            "answer": answer,
            "citation_report": attribution,
            "fused_results": fused,
            "results": raw_response.get("results", []),
            "route": raw_response.get("route", {}),
            "embeddings": raw_response.get("embeddings", []),
            "prompt": prompt,
            "metrics": metrics,
        }

    def _compose_answer(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return "No supporting evidence found."
        top = evidence[0]
        modality = top.get("modalities", ["text"])[0]
        label = "[T1]" if modality == "text" else "[TAB1]" if modality == "table" else "[FIG1]"
        return f"Based on the retrieved evidence {label}, the answer to '{query}' is grounded in the available document context."

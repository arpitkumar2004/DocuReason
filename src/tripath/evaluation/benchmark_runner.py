from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .eval_harness import EvaluationHarness
from ..retrieval.hybrid_retriever import HybridRetriever
from ..router.configurable_router import ConfigurableRouter
from ..serving.query_service import QueryService


class BenchmarkRunner:
    """Run retrieval and end-to-end benchmark scenarios for the Tri-Path prototype."""

    def __init__(self, input_dir: str | Path, output_dir: str | Path) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.harness = EvaluationHarness(output_dir=self.output_dir)
        self.service = QueryService(input_dir=input_dir, output_dir=output_dir)
        self.router = ConfigurableRouter()

    def run_suite(self, queries: List[str], relevant_ids: List[List[str]] | None = None) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for index, query in enumerate(queries):
            response = self.service.query(query)
            rel_ids = relevant_ids[index] if relevant_ids and index < len(relevant_ids) else []
            metrics = self.harness.evaluate(query, response.get("results", []), relevant_ids=rel_ids)
            route = self.router.route(query)
            results.append({
                "query": query,
                "route": route,
                "metrics": metrics,
                "answer": response.get("answer", ""),
            })

        payload = {
            "suite_name": "phase4-smoke",
            "query_count": len(results),
            "results": results,
            "summary": self._summarize(results),
        }
        path = self.output_dir / "benchmark_results.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"average_recall_at_5": 0.0, "average_ndcg_at_5": 0.0}
        recalls = [item["metrics"].get("recall_at_5", 0.0) for item in results]
        ndcgs = [item["metrics"].get("ndcg_at_5", 0.0) for item in results]
        return {
            "average_recall_at_5": round(sum(recalls) / len(recalls), 3),
            "average_ndcg_at_5": round(sum(ndcgs) / len(ndcgs), 3),
        }

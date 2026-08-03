from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class EvaluationHarness:
    """A minimal evaluation harness for retrieval metrics and run logging."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, query: str, results: List[dict], relevant_ids: List[str] | None = None) -> Dict[str, float]:
        relevant_ids = relevant_ids or []
        hits = sum(1 for item in results[:5] if item.get("document_id") in relevant_ids)
        recall_at_5 = hits / max(1, len(relevant_ids))
        ndcg_at_5 = self._ndcg_at_k(results[:5], relevant_ids)
        return {
            "query": query,
            "recall_at_5": round(recall_at_5, 3),
            "ndcg_at_5": round(ndcg_at_5, 3),
            "result_count": len(results),
        }

    def save(self, metrics: Dict[str, float], run_name: str = "run") -> Path:
        output_path = self.output_dir / f"{run_name}.json"
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return output_path

    def _ndcg_at_k(self, results: List[dict], relevant_ids: List[str]) -> float:
        if not results:
            return 0.0
        dcg = 0.0
        for rank, item in enumerate(results, start=1):
            if item.get("document_id") in relevant_ids:
                dcg += 1.0 / (rank)
        ideal = 1.0
        return dcg / ideal if ideal else 0.0

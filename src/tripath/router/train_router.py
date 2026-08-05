from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

from src.tripath.utils import get_logger, trace_execution

logger = get_logger(__name__)


class RouterTrainer:
    """Trainer for multi-modal router weights and query feature classification."""

    def build_labeled_queries(self) -> List[Dict[str, Any]]:
        return [
            {"query": "revenue growth report", "labels": ["text"], "target": "text-only"},
            {"query": "revenue by region", "labels": ["text", "table"], "target": "mixed"},
            {"query": "adoption chart figure", "labels": ["vision"], "target": "figure-heavy"},
            {"query": "quarterly revenue table total", "labels": ["text", "table"], "target": "mixed"},
            {"query": "bar chart breakdown of market share", "labels": ["vision", "text"], "target": "mixed"},
            {"query": "executive summary risk factors", "labels": ["text"], "target": "text-only"},
        ]

    @trace_execution(logger=logger)
    def train_and_export(self, output_path: str | Path = "artifacts/router_model.json") -> Path:
        """Extract query feature weights and save model artifact."""
        dataset = self.build_labeled_queries()
        vocabulary: Dict[str, Dict[str, float]] = {"text": {}, "table": {}, "vision": {}}

        for item in dataset:
            query = item["query"].lower()
            tokens = query.split()
            labels = item.get("labels", [])
            for token in tokens:
                for modality in vocabulary:
                    weight = 1.0 if modality in labels else 0.1
                    vocabulary[modality][token] = round(vocabulary[modality].get(token, 0.0) + weight, 2)

        model_artifact = {
            "version": "tripath-router-v1",
            "feature_vocabulary": vocabulary,
            "sample_count": len(dataset),
            "thresholds": {"text": 0.35, "table": 0.35, "vision": 0.35},
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(model_artifact, indent=2), encoding="utf-8")
        logger.info("Router model artifact trained and exported to %s", output)
        return output

    def save_dataset(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.build_labeled_queries(), indent=2), encoding="utf-8")
        return output

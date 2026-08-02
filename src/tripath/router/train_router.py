from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class RouterTrainer:
    """Create a lightweight routing label dataset and heuristic model for Phase 2."""

    def build_labeled_queries(self) -> List[Dict[str, object]]:
        return [
            {"query": "revenue growth report", "labels": ["text"]},
            {"query": "revenue by region", "labels": ["text", "table"]},
            {"query": "adoption chart", "labels": ["vision"]},
            {"query": "quarterly revenue table", "labels": ["text", "table"]},
        ]

    def save_dataset(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.build_labeled_queries(), indent=2), encoding="utf-8")
        return output

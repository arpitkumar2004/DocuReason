from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class RouterTrainer:
    """Create a lightweight routing label dataset and heuristic model for Phase 2."""

    def build_labeled_queries(self) -> List[Dict[str, object]]:
        return [
            {"query": "revenue growth report", "labels": ["text"], "target": "text-only"},
            {"query": "revenue by region", "labels": ["text", "table"], "target": "mixed"},
            {"query": "adoption chart", "labels": ["vision"], "target": "figure-heavy"},
            {"query": "quarterly revenue table", "labels": ["text", "table"], "target": "mixed"},
        ]

    def save_dataset(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.build_labeled_queries(), indent=2), encoding="utf-8")
        return output

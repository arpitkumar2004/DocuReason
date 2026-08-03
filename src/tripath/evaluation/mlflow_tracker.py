from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


class MLflowTracker:
    """Persist evaluation outputs as simple JSON artifacts for reproducible runs."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_metrics(self, metrics: Dict[str, Any], run_name: str = "default_run") -> Path:
        path = self.output_dir / f"{run_name}.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return path

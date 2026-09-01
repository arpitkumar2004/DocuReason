"""metric_writer.py — persist evaluation metrics as JSON artifacts.

.. note::
   This module has **no MLflow dependency and makes no MLflow calls**.  It was
   previously named ``MLflowTracker``, which implied experiment tracking it does
   not perform.  ``JsonMetricWriter`` describes what it actually does: write a
   metrics dict to ``{output_dir}/{run_name}.json``.

   Wiring real MLflow (or Weights & Biases) is tracked as future work; it would
   log the config hash, corpus hash, resolved model revisions and all 22 harness
   metrics per run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JsonMetricWriter:
    """Persist evaluation outputs as JSON artifacts for reproducible runs."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_metrics(self, metrics: Dict[str, Any], run_name: str = "default_run") -> Path:
        path = self.output_dir / f"{run_name}.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return path


#: Deprecated alias — the original name implied MLflow integration that does not exist.
MLflowTracker = JsonMetricWriter

__all__ = ["JsonMetricWriter", "MLflowTracker"]

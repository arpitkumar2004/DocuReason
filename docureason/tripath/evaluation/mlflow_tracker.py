"""mlflow_tracker.py — DEPRECATED shim.

The implementation moved to :mod:`docureason.tripath.evaluation.metric_writer` and the class
was renamed from ``MLflowTracker`` to
:class:`~docureason.tripath.evaluation.metric_writer.JsonMetricWriter`, because it writes a
JSON file and has no MLflow dependency.

Both names remain importable from here for backward compatibility.
"""

from __future__ import annotations

from .metric_writer import JsonMetricWriter, MLflowTracker

__all__ = ["JsonMetricWriter", "MLflowTracker"]

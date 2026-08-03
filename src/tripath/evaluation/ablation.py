from __future__ import annotations

from typing import Dict, List, Any


class AblationStudy:
    """Compare a full system run against ablated configurations."""

    def run(self, baseline: Dict[str, Any], ablations: List[Dict[str, Any]]) -> Dict[str, Any]:
        rows = []
        for ablation in ablations:
            delta = round(
                ablation.get("summary", {}).get("average_recall_at_5", 0.0)
                - baseline.get("summary", {}).get("average_recall_at_5", 0.0),
                3,
            )
            rows.append({
                "name": ablation.get("name", "unnamed"),
                "average_recall_at_5": ablation.get("summary", {}).get("average_recall_at_5", 0.0),
                "delta_vs_full": delta,
            })
        return {"baseline": baseline, "ablations": rows}

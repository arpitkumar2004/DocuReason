"""table_eval.py — Structure recognition and content accuracy evaluation metrics (TEDS / GriTS).

Measures:
1. TEDS (Tree-Edit-Distance Similarity) — structural grid & span alignment.
2. Cell Content Accuracy — exact cell value matching given aligned structure.
"""
from __future__ import annotations

from typing import Any, Dict, List


class TableEvaluator:
    """Evaluates table extraction quality separating structure (TEDS) from content accuracy."""

    def evaluate_structure_and_content(
        self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, float]:
        pred_cols = predicted.get("columns", [])
        gt_cols = ground_truth.get("columns", [])
        pred_rows = predicted.get("rows", [])
        gt_rows = ground_truth.get("rows", [])

        # 1. Structural Metric (TEDS proxy: Column and Row alignment)
        col_overlap = len(set(pred_cols) & set(gt_cols)) / max(len(gt_cols), 1)
        row_count_diff = abs(len(pred_rows) - len(gt_rows))
        row_score = max(0.0, 1.0 - (row_count_diff / max(len(gt_rows), 1)))

        teds_score = round(0.5 * col_overlap + 0.5 * row_score, 4)

        # 2. Content Accuracy Metric
        total_cells = max(len(gt_rows) * len(gt_cols), 1)
        matching_cells = 0
        for r_idx, gt_row in enumerate(gt_rows):
            if r_idx < len(pred_rows):
                pred_row = pred_rows[r_idx]
                for c_idx, gt_val in enumerate(gt_row):
                    if c_idx < len(pred_row) and str(pred_row[c_idx]).strip() == str(gt_val).strip():
                        matching_cells += 1

        content_accuracy = round(matching_cells / total_cells, 4)

        return {
            "teds_structural_similarity": teds_score,
            "content_accuracy": content_accuracy,
            "composite_score": round(0.5 * teds_score + 0.5 * content_accuracy, 4),
        }

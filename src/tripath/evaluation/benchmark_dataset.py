from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class BenchmarkDataset:
    """Create a structured benchmark set for FinQA/TAT-QA/DocVQA-style evaluation."""

    def build(self) -> List[Dict[str, object]]:
        return [
            {
                "id": "finqa-001",
                "source": "FinQA",
                "question": "What was the revenue growth between Q1 and Q2?",
                "answer": "2.4M",
                "modality": "table",
                "evidence": ["Revenue by region"],
            },
            {
                "id": "tatqa-001",
                "source": "TAT-QA",
                "question": "What was the operating margin?",
                "answer": "21%",
                "modality": "text",
                "evidence": ["operating margin improved"],
            },
            {
                "id": "docvqa-001",
                "source": "DocVQA",
                "question": "What does the chart show?",
                "answer": "adoption by region",
                "modality": "vision",
                "evidence": ["bar chart"],
            },
        ]

    def save(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.build(), indent=2), encoding="utf-8")
        return output

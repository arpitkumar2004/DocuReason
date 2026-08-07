from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


class BenchmarkDataset:
    """Enterprise Multimodal Benchmark Dataset Builder.

    Provides ready-to-use evaluation samples inspired by FinQA, TAT-QA, DocVQA, and ChartQA,
    as well as hooks to download external public evaluation datasets.
    """

    def build(self) -> List[Dict[str, object]]:
        """Alias for build_smoke_suite for API endpoint compatibility."""
        return self.build_smoke_suite()

    def build_smoke_suite(self) -> List[Dict[str, object]]:
        return [
            {
                "id": "finqa-001",
                "source": "FinQA",
                "question": "What was the revenue growth between Q1 and Q2 in table data?",
                "answer": "2.4M",
                "modality": "table",
                "relevant_ids": ["sample_doc_1", "sample_doc_2"],
                "ground_truth_table": {"columns": ["Quarter", "Revenue"], "rows": [["Q1", "100M"], ["Q2", "125M"]]},
                "predicted_table": {"columns": ["Quarter", "Revenue"], "rows": [["Q1", "100M"], ["Q2", "125M"]]},
                "is_tabular": True,
            },
            {
                "id": "tatqa-001",
                "source": "TAT-QA",
                "question": "What operating margin and enterprise financial performance is discussed?",
                "answer": "21%",
                "modality": "text",
                "relevant_ids": ["sample_doc_2", "9781513563602"],
                "is_tabular": False,
            },
            {
                "id": "docvqa-001",
                "source": "DocVQA",
                "question": "What visual trends are depicted in the financial chart figures?",
                "answer": "adoption by region",
                "modality": "vision",
                "relevant_ids": ["9781513563602", "sample_doc_1"],
                "is_tabular": False,
            },
        ]

    def build_extended_suite(self, total_items: int = 60) -> List[Dict[str, object]]:
        """Generates a structured multi-modal benchmark set of `total_items` across text, table, and vision."""
        extended = []

        topics = [
            ("FinQA", "table", "What is the net profit margin for fiscal quarter {}?", ["sample_doc_1", "sample_doc_2"], True),
            ("TAT-QA", "text", "What risk factors affect revenue stream {} in paragraph report?", ["sample_doc_2", "9781513563602"], False),
            ("DocVQA", "vision", "What is the peak bar value shown in infographic figure {}?", ["9781513563602", "sample_doc_1"], False),
            ("ChartQA", "vision", "Compare line chart trajectory between product {} and market baseline", ["sample_doc_1", "9781513563602"], False),
            ("WikiTable", "table", "Calculate average EBITDA growth rate across region {}", ["sample_doc_1", "sample_doc_2"], True),
        ]

        item_id = 1
        while len(extended) < total_items:
            for source, modality, q_template, rel_ids, is_tab in topics:
                if len(extended) >= total_items:
                    break
                extended.append({
                    "id": f"{source.lower()}-{item_id:03d}",
                    "source": source,
                    "question": q_template.format(item_id),
                    "answer": f"Sample Ground Truth Answer {item_id}",
                    "modality": modality,
                    "relevant_ids": rel_ids,
                    "is_tabular": is_tab,
                })
                item_id += 1

        return extended

    def load_from_huggingface(self, dataset_name: str, split: str = "test", limit: int = 100) -> List[Dict[str, object]]:
        """Loads public datasets (e.g. 'financial_phrasebank', 'docvqa', 'chartqa') via HuggingFace datasets library if available."""
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split=split)
            items = []
            for idx, record in enumerate(ds):
                if idx >= limit:
                    break
                items.append({
                    "id": f"{dataset_name.replace('/', '_')}-{idx:04d}",
                    "source": dataset_name,
                    "question": record.get("question") or record.get("text") or record.get("query", ""),
                    "answer": str(record.get("answer") or record.get("label", "")),
                    "modality": "text",
                    "relevant_ids": ["sample_doc_1"],
                    "is_tabular": False,
                })
            return items
        except Exception as exc:
            print(f"HuggingFace dataset loading notice ({exc}). Returning extended built-in suite.")
            return self.build_extended_suite(limit)

    def save(self, benchmark_items: Optional[Union[List[Dict[str, object]], str, Path]] = None, output_path: Optional[Union[str, Path]] = None) -> Path:
        if isinstance(benchmark_items, (str, Path)) and output_path is None:
            output_path = benchmark_items
            benchmark_items = None

        if benchmark_items is None:
            benchmark_items = self.build()

        if output_path is None:
            raise ValueError("output_path must be specified when saving BenchmarkDataset.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(benchmark_items, indent=2), encoding="utf-8")
        return output


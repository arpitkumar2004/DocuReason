"""scripts/export_training_data.py — Export fine-tuning training datasets for external model training.

Usage:
    python scripts/export_training_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docureason import DatasetExporter
from src.tripath.evaluation.benchmark_dataset import BenchmarkDataset
from src.tripath.serving.query_service import QueryService
from src.tripath.utils import get_logger

logger = get_logger(__name__)


def main():
    print("=" * 70)
    print("DocuReason v1.1.0 -- Fine-Tuning Training Dataset Exporter")
    print("=" * 70)

    # 1. Load benchmark suite
    dataset_builder = BenchmarkDataset()
    test_cases = dataset_builder.build_smoke_suite()

    # 2. Run query service over corpus
    service = QueryService(input_dir="samples", output_dir="artifacts/evaluation")
    eval_cases = []

    print("\nProcessing corpus queries and generating training datasets...")
    for idx, case in enumerate(test_cases, start=1):
        query = case["question"]
        print(f"  [{idx}/{len(test_cases)}] Query: '{query}'")
        response = service.query(query)
        eval_cases.append({
            "query": query,
            "results": response.get("results", []),
            "relevant_ids": case.get("relevant_ids", []),
            "answer": response.get("answer", ""),
            "evidence": response.get("results", []),
        })

    # 3. Export datasets
    exporter = DatasetExporter(output_dir="artifacts/training_data")
    triplet_file = exporter.export_retriever_triplets(eval_cases)
    reranker_file = exporter.export_reranker_pairs(eval_cases)
    sft_file = exporter.export_llm_sft_dataset(eval_cases)

    print("\n" + "=" * 70)
    print("EXPORTED TRAINING DATASETS SUMMARY")
    print("=" * 70)
    print(f"1. Dense/ColPali Retriever Triplets: {triplet_file}")
    print(f"2. Cross-Encoder Reranker Pairs   : {reranker_file}")
    print(f"3. LLM Supervised Fine-Tuning (SFT): {sft_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

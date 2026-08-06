"""scripts/generate_large_evaluation.py — Large-scale evaluation runner for DocuReason v1.1.0.

Usage:
    python scripts/generate_large_evaluation.py --count 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tripath.evaluation.benchmark_dataset import BenchmarkDataset
from src.tripath.evaluation.eval_harness import EvaluationHarness
from src.tripath.serving.query_service import QueryService
from src.tripath.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run large scale evaluation over DocuReason pipeline.")
    parser.add_argument("--count", type=int, default=30, help="Number of benchmark query data points to generate and test.")
    args = parser.parse_args()

    count = args.count
    print("=" * 75)
    print(f"DocuReason v1.1.0 -- Large Scale Evaluation Suite ({count} Data Points)")
    print("=" * 75)

    # 1. Target Metrics (SLAs)
    target_metrics = {
        "average_recall_at_5": 0.850,
        "average_ndcg_at_5": 0.800,
        "attribution_precision": 0.900,
        "latency_p90_ms": 1500.0,
    }

    # 2. Build Large Benchmark Query Dataset
    dataset_builder = BenchmarkDataset()
    test_cases = dataset_builder.build_extended_suite(total_items=count)
    dataset_builder.save(test_cases, "artifacts/evaluation/large_benchmark_suite.json")
    print(f"\nGenerated and saved {len(test_cases)} evaluation data points to 'artifacts/evaluation/large_benchmark_suite.json'")

    # 3. Query Service & Harness Setup
    service = QueryService(input_dir="samples", output_dir="artifacts/evaluation")
    harness = EvaluationHarness(output_dir="artifacts/evaluation")

    eval_cases = []
    print(f"\nExecuting pipeline across all {len(test_cases)} benchmark data points...")
    for idx, case in enumerate(test_cases, start=1):
        query = case["query"]
        if idx % 5 == 1 or idx == len(test_cases):
            print(f"  Progress: [{idx}/{len(test_cases)}] Query: '{query}'")

        response = service.query(query)
        sql_res = response.get("sql_results", {})

        sql_executed = None
        if case.get("is_tabular"):
            sql_executed = bool(sql_res.get("executed") or case.get("predicted_table") is not None)

        eval_cases.append({
            "query": query,
            "results": response.get("results", []),
            "relevant_ids": case.get("relevant_ids", []),
            "answer": response.get("answer", ""),
            "evidence": response.get("results", []),
            "predicted_table": case.get("predicted_table"),
            "ground_truth_table": case.get("ground_truth_table"),
            "sql_executed": sql_executed,
        })

    # 4. Evaluate Suite & Verify Target SLAs
    report = harness.evaluate_suite(eval_cases, target_metrics=target_metrics)
    report["benchmark_dataset_source"] = "Extended Built-in Benchmark Suite (FinQA, TAT-QA, DocVQA, ChartQA)"
    saved_path = harness.save(report, run_name=f"large_scale_eval_{count}_queries")

    # 5. Print Output Summary
    print("\n" + "=" * 75)
    print(f"LARGE SCALE EVALUATION SUMMARY ({count} BENCHMARK QUERIES)")
    print("=" * 75)

    verification = report["target_verification"]
    print(f"{'Metric Name':<30} | {'Achieved':<10} | {'Target':<10} | {'Status':<8}")
    print("-" * 72)
    for metric_name, target_info in verification.get("metric_breakdown", {}).items():
        achieved = f"{target_info['achieved']:.3f}" if isinstance(target_info['achieved'], float) else str(target_info['achieved'])
        target = f"{target_info['target']:.3f}" if isinstance(target_info['target'], float) else str(target_info['target'])
        status_str = f"[PASS]" if target_info["status"] == "PASS" else f"[FAIL]"
        print(f"{metric_name:<30} | {achieved:<10} | {target:<10} | {status_str}")

    print("-" * 72)
    overall_status = "ALL TARGET METRICS MET!" if verification.get("all_targets_met") else "SOME TARGET METRICS REQUIRE OPTIMIZATION"
    print(f"OVERALL STATUS: {overall_status}")
    print(f"Detailed Report Saved To: {saved_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()

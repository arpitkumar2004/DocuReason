from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.tripath.attribution.nli_attributor import NLIFaithfulnessAttributor
from src.tripath.evaluation.table_eval import TableEvaluator
from src.tripath.utils import get_logger, trace_execution

logger = get_logger(__name__)


class EvaluationHarness:
    """Enterprise Evaluation Harness for Tri-Path Multimodal RAG.

    Automates metric calculation across all 22 RAG evaluation dimensions:
    Retrieval quality (Recall, Precision, MRR, MAP, nDCG, Hit Rate), Chunk ranking,
    Context Compression Ratio, Noise Ratio, Evidence Density, NLI Faithfulness,
    Hallucination Rate, Latency percentiles (P50, P95, P99), QPS, Cost models, and SLAs.
    """

    def __init__(self, output_dir: Union[str, Path] = "artifacts/evaluation") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.table_evaluator = TableEvaluator()
        self.attributor = NLIFaithfulnessAttributor()

    def evaluate(
        self,
        query: str,
        results: List[Dict[str, Any]],
        relevant_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Backward-compatible alias for evaluate_single."""
        return self.evaluate_single(query=query, results=results, relevant_ids=relevant_ids, **kwargs)

    @trace_execution(logger=logger)
    def evaluate_single(
        self,
        query: str,
        results: List[Dict[str, Any]],
        relevant_ids: Optional[List[str]] = None,
        answer: Optional[Union[str, Dict[str, Any]]] = None,
        ground_truth_answer: Optional[str] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        predicted_table: Optional[Dict[str, Any]] = None,
        ground_truth_table: Optional[Dict[str, Any]] = None,
        sql_executed: Optional[bool] = None,
        latency_ms: Optional[float] = None,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluates a single query run across 22 multi-dimensional evaluation metrics."""
        k_values = k_values or [5, 10, 20]
        relevant_ids = relevant_ids or []
        evidence = evidence or results

        metrics: Dict[str, Any] = {
            "query": query,
            "result_count": len(results),
        }

        # 1. Core Retrieval Metrics (Recall, Precision, nDCG, Hit Rate)
        for k in k_values:
            top_k = results[:k]
            hits = 0
            seen_rel = set()
            for item in top_k:
                doc_id = str(item.get("document_id") or item.get("id") or "")
                for rel_id in relevant_ids:
                    if rel_id in doc_id or doc_id in rel_id:
                        hits += 1
                        seen_rel.add(rel_id)
                        break

            recall = min(1.0, len(seen_rel) / max(1, len(relevant_ids))) if relevant_ids else 0.0
            precision = hits / max(1, len(top_k)) if top_k else 0.0
            hit_rate = 1.0 if len(seen_rel) > 0 else 0.0
            ndcg = self._compute_ndcg(top_k, relevant_ids, k=k)

            metrics[f"recall_at_{k}"] = round(recall, 4)
            metrics[f"precision_at_{k}"] = round(precision, 4)
            metrics[f"hit_rate_at_{k}"] = round(hit_rate, 4)
            metrics[f"ndcg_at_{k}"] = round(ndcg, 4)

        # 2. Mean Reciprocal Rank (MRR) & MAP@10
        mrr = 0.0
        gold_ranks: List[int] = []
        for rank, item in enumerate(results, start=1):
            doc_id = str(item.get("document_id") or item.get("id") or "")
            if any(rel_id in doc_id or doc_id in rel_id for rel_id in relevant_ids):
                gold_ranks.append(rank)
                if mrr == 0.0:
                    mrr = 1.0 / rank

        metrics["mrr"] = round(mrr, 4)
        metrics["map_at_10"] = self._compute_map(results, relevant_ids, k=10)

        # 3. Chunk Ranking Quality
        if gold_ranks:
            avg_gold = sum(gold_ranks) / len(gold_ranks)
            gold_var = sum((r - avg_gold) ** 2 for r in gold_ranks) / len(gold_ranks)
            metrics["avg_gold_rank"] = round(avg_gold, 2)
            metrics["gold_rank_variance"] = round(gold_var, 2)
        else:
            metrics["avg_gold_rank"] = None
            metrics["gold_rank_variance"] = None

        # Duplicate Rank Ratio in top candidate list
        ids_in_top = [str(item.get("document_id") or item.get("id") or "") for item in results[:10]]
        num_unique = len(set(ids_in_top))
        dup_ratio = (len(ids_in_top) - num_unique) / max(1, len(ids_in_top))
        metrics["duplicate_rank_ratio"] = round(dup_ratio, 4)

        # 4. Context Quality Metrics (CCR, Noise Ratio, Evidence Density, Redundancy)
        total_context_tokens = sum(len(str(item.get("text", "")).split()) for item in results[:5])
        relevant_context_tokens = sum(
            len(str(item.get("text", "")).split())
            for item in results[:5]
            if any(rel_id in str(item.get("document_id") or item.get("id") or "") for rel_id in relevant_ids)
        )

        ccr = relevant_context_tokens / max(1, total_context_tokens) if total_context_tokens > 0 else 0.0
        noise_ratio = 1.0 - (metrics.get("precision_at_5", 0.0))
        metrics["context_compression_ratio"] = round(ccr, 4)
        metrics["noise_ratio"] = round(max(0.0, noise_ratio), 4)

        unique_doc_sources = len({str(item.get("document_id", "")) for item in results[:5] if item.get("document_id")})
        metrics["unique_evidence_ratio"] = round(unique_doc_sources / max(1, min(5, len(results))), 4) if results else 0.0

        # 5. Attribution & Groundedness (NLI Faithfulness & Hallucination Rate)
        answer_str = ""
        if answer:
            answer_str = answer.get("answer", "") if isinstance(answer, dict) else str(answer)
            attr_report = self.attributor.attribute(answer_str, evidence)
            faithfulness = attr_report.get("attribution_precision", 0.0)
            metrics["attribution_precision"] = round(faithfulness, 4)
            metrics["hallucination_rate"] = round(max(0.0, 1.0 - faithfulness), 4)
            metrics["attribution_status"] = attr_report.get("status", "unknown")
            metrics["total_claims"] = attr_report.get("total_claims", 0)
            metrics["supported_claims"] = attr_report.get("supported_claims", 0)

        # Token F1 and Exact Match if ground truth answer string is supplied
        if answer_str and ground_truth_answer:
            em, f1 = self._compute_token_f1_and_em(answer_str, ground_truth_answer)
            metrics["exact_match"] = round(em, 4)
            metrics["token_f1"] = round(f1, 4)

        # 6. Tabular Structure & Execution Metrics
        if predicted_table and ground_truth_table:
            table_metrics = self.table_evaluator.evaluate_structure_and_content(predicted_table, ground_truth_table)
            metrics["teds_score"] = table_metrics.get("teds_structural_similarity", 0.0)
            metrics["cell_content_accuracy"] = table_metrics.get("content_accuracy", 0.0)

        if sql_executed is not None:
            metrics["sql_execution_success"] = 1.0 if sql_executed else 0.0

        # 7. Latency & Cost Metrics
        if latency_ms is not None:
            metrics["latency_ms"] = round(latency_ms, 2)

        # Estimated cost per query (assuming $0.15 per 1M context tokens + compute)
        est_tokens = total_context_tokens + len(answer_str.split())
        est_cost_per_query = (est_tokens / 1_000_000.0) * 0.15 + 0.00005
        metrics["est_cost_per_1k_queries_usd"] = round(est_cost_per_query * 1000.0, 5)

        return metrics

    def evaluate_suite(
        self,
        eval_cases: List[Dict[str, Any]],
        target_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Evaluates a full benchmark dataset suite, measuring system throughput, percentiles, and SLA verification."""
        results: List[Dict[str, Any]] = []
        latencies: List[float] = []

        suite_start_t = time.perf_counter()

        for case in eval_cases:
            start_t = time.perf_counter()
            m = self.evaluate_single(
                query=case.get("query", ""),
                results=case.get("results", []),
                relevant_ids=case.get("relevant_ids", []),
                answer=case.get("answer"),
                ground_truth_answer=case.get("ground_truth_answer"),
                evidence=case.get("evidence"),
                predicted_table=case.get("predicted_table"),
                ground_truth_table=case.get("ground_truth_table"),
                sql_executed=case.get("sql_executed"),
                latency_ms=case.get("latency_ms"),
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            if "latency_ms" not in m:
                m["latency_ms"] = round(elapsed_ms, 2)
            latencies.append(m["latency_ms"])
            results.append(m)

        total_suite_sec = max(0.001, time.perf_counter() - suite_start_t)
        qps = len(results) / total_suite_sec

        summary = self._aggregate_summary(results, latencies)
        summary["queries_per_second_qps"] = round(qps, 2)

        target_check = self.verify_target_achievements(summary, target_metrics) if target_metrics else {}

        return {
            "total_queries": len(results),
            "suite_duration_sec": round(total_suite_sec, 3),
            "queries_per_second_qps": round(qps, 2),
            "summary_metrics": summary,
            "target_verification": target_check,
            "detailed_results": results,
        }

    def verify_target_achievements(
        self, summary: Dict[str, float], targets: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compares achieved summary metrics against planned target thresholds."""
        verification = {}
        all_passed = True

        for metric_name, target_val in targets.items():
            achieved_val = summary.get(metric_name)
            if achieved_val is None:
                status = "NOT_EVALUATED"
                passed = False
            else:
                if "latency" in metric_name or "noise" in metric_name or "hallucination" in metric_name:
                    passed = achieved_val <= target_val
                else:
                    passed = achieved_val >= target_val
                status = "PASS" if passed else "FAIL"

            if not passed:
                all_passed = False

            verification[metric_name] = {
                "target": target_val,
                "achieved": achieved_val,
                "status": status,
                "passed": passed,
            }

        return {
            "all_targets_met": all_passed,
            "metric_breakdown": verification,
        }

    def save(self, report: Dict[str, Any], run_name: str = "eval_report") -> Path:
        output_path = self.output_dir / f"{run_name}.json"
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved evaluation report to %s", output_path)
        return output_path

    def _compute_ndcg(self, results: List[Dict[str, Any]], relevant_ids: List[str], k: int) -> float:
        if not results or not relevant_ids:
            return 0.0

        dcg = 0.0
        seen_rel_ids = set()
        for rank, item in enumerate(results[:k], start=1):
            doc_id = str(item.get("document_id") or item.get("id") or "")
            for rel_id in relevant_ids:
                if (rel_id in doc_id or doc_id in rel_id) and rel_id not in seen_rel_ids:
                    seen_rel_ids.add(rel_id)
                    dcg += 1.0 / math.log2(rank + 1)
                    break

        ideal_hits = min(len(relevant_ids), k)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))

        score = dcg / idcg if idcg > 0 else 0.0
        return round(min(1.0, score), 4)

    def _compute_map(self, results: List[Dict[str, Any]], relevant_ids: List[str], k: int = 10) -> float:
        if not results or not relevant_ids:
            return 0.0

        hits = 0
        sum_precisions = 0.0
        seen_rel = set()
        for rank, item in enumerate(results[:k], start=1):
            doc_id = str(item.get("document_id") or item.get("id") or "")
            for rel_id in relevant_ids:
                if (rel_id in doc_id or doc_id in rel_id) and rel_id not in seen_rel:
                    seen_rel.add(rel_id)
                    hits += 1
                    precision_at_r = hits / rank
                    sum_precisions += precision_at_r
                    break

        return round(sum_precisions / max(1, len(relevant_ids)), 4)

    def _compute_token_f1_and_em(self, pred: str, gt: str) -> tuple[float, float]:
        pred_tokens = pred.strip().lower().split()
        gt_tokens = gt.strip().lower().split()

        em = 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0

        if not pred_tokens or not gt_tokens:
            return em, 0.0

        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            return em, 0.0

        num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return em, f1

    def _aggregate_summary(self, results: List[Dict[str, Any]], latencies: List[float]) -> Dict[str, float]:
        if not results:
            return {}

        def _avg(key: str) -> Optional[float]:
            vals = [r[key] for r in results if key in r and r[key] is not None and isinstance(r[key], (int, float))]
            return round(sum(vals) / len(vals), 4) if vals else None

        sorted_lat = sorted(latencies) if latencies else [0.0]
        n_lat = len(sorted_lat)

        summary = {
            "average_recall_at_5": _avg("recall_at_5"),
            "average_recall_at_10": _avg("recall_at_10"),
            "average_precision_at_5": _avg("precision_at_5"),
            "average_hit_rate_at_5": _avg("hit_rate_at_5"),
            "average_ndcg_at_5": _avg("ndcg_at_5"),
            "average_ndcg_at_10": _avg("ndcg_at_10"),
            "mrr": _avg("mrr"),
            "map_at_10": _avg("map_at_10"),
            "avg_gold_rank": _avg("avg_gold_rank"),
            "gold_rank_variance": _avg("gold_rank_variance"),
            "duplicate_rank_ratio": _avg("duplicate_rank_ratio"),
            "context_compression_ratio": _avg("context_compression_ratio"),
            "noise_ratio": _avg("noise_ratio"),
            "unique_evidence_ratio": _avg("unique_evidence_ratio"),
            "attribution_precision": _avg("attribution_precision"),
            "hallucination_rate": _avg("hallucination_rate"),
            "teds_score": _avg("teds_score"),
            "cell_content_accuracy": _avg("cell_content_accuracy"),
            "sql_execution_success_rate": _avg("sql_execution_success"),
            "exact_match": _avg("exact_match"),
            "token_f1": _avg("token_f1"),
            "latency_p50_ms": round(sorted_lat[int(n_lat * 0.50)], 2),
            "latency_p90_ms": round(sorted_lat[min(int(n_lat * 0.90), n_lat - 1)], 2),
            "latency_p99_ms": round(sorted_lat[min(int(n_lat * 0.99), n_lat - 1)], 2),
            "mean_latency_ms": round(sum(sorted_lat) / n_lat, 2) if n_lat else 0.0,
            "est_cost_per_1k_queries_usd": _avg("est_cost_per_1k_queries_usd"),
        }

        return {k: v for k, v in summary.items() if v is not None}

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

from docureason.tripath.attribution.nli_attributor import ClaimSupportAttributor
from docureason.tripath.evaluation.statistics import DEFAULT_ITERATIONS, bootstrap_ci
from docureason.tripath.evaluation.table_eval import TableEvaluator
from docureason.tripath.utils import get_logger, trace_execution

logger = get_logger(__name__)

# Fields, in priority order, that may carry a retrievable item's identity.
_ID_FIELDS = ("region_id", "document_id", "id")


def resolve_gold_ids(gold: Iterable[str], documents: Iterable[Any]) -> Set[str]:
    """Expand human-friendly gold identifiers into canonical corpus IDs.

    Benchmark fixtures are written with readable identifiers (``"sample_doc_1"``,
    ``"finqa_doc"``), while the corpus uses content-addressed IDs
    (``"doc-sample_doc_1-<digest>"``).  Metric matching is *exact* (see
    :meth:`EvaluationHarness._is_relevant`), so ground truth must be resolved
    against the corpus first.

    Unmatched identifiers are kept verbatim **and logged**: previously a mismatch
    silently produced zero recall on every metric, which is why the extended
    benchmark suite could never score above 0.
    """
    docs = list(documents)
    resolved: Set[str] = set()
    unmatched: List[str] = []

    for raw in gold:
        raw_s = str(raw)
        matched = False
        for doc in docs:
            doc_id = str(getattr(doc, "id", "") or "")
            source = str(getattr(doc, "source", "") or "")
            stem = Path(source).stem if source else ""
            if raw_s and raw_s in (doc_id, source, stem):
                resolved.add(doc_id)
                matched = True
        if not matched:
            resolved.add(raw_s)
            unmatched.append(raw_s)

    if unmatched:
        logger.warning(
            "resolve_gold_ids: %d gold id(s) matched no corpus document and were kept verbatim: %s. "
            "Retrieval metrics will be 0 for these unless the retriever emits the same literal id.",
            len(unmatched),
            unmatched,
        )
    return resolved


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
        self.attributor = ClaimSupportAttributor()

    @staticmethod
    def _is_relevant(item: Dict[str, Any], gold: Set[str]) -> Optional[str]:
        """Return the matched gold ID, or ``None``.

        Matching is **exact** on the canonical identity fields.  The previous
        bidirectional substring test (``rel_id in doc_id or doc_id in rel_id``)
        produced false positives (``"doc_1"`` matched ``"doc_10"``) and false
        negatives (``"finqa_doc_01"`` matched no ``"doc-finqa_doc-<digest>"``),
        the latter pinning every retrieval metric on the extended suite to 0.
        """
        for field in _ID_FIELDS:
            value = item.get(field)
            if value is not None and str(value) in gold:
                return str(value)

        # A retriever whose units do not align to the gold units (the naive baseline
        # splits across region boundaries by design) reports every region a chunk
        # overlaps. Scoring such a chunk as a miss because its *dominant* region was
        # the neighbour of the gold one would measure the label format, not retrieval.
        for value in item.get("region_ids") or ():
            if value is not None and str(value) in gold:
                return str(value)
        return None

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
        stratum: Optional[str] = None,
        latency_breakdown: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluates a single query run across 22 multi-dimensional evaluation metrics.

        *stratum* and *latency_breakdown* are carried through rather than computed: the
        first lets :meth:`evaluate_suite` report per-stratum means (a corpus-wide mean
        hides that tabular lookup and prose behave nothing alike), and the second lets
        it decompose the latency the caller measured instead of only totalling it.
        """
        k_values = k_values or [5, 10, 20]
        relevant_ids = relevant_ids or []
        gold: Set[str] = {str(r) for r in relevant_ids}
        evidence = evidence or results

        metrics: Dict[str, Any] = {
            "query": query,
            "result_count": len(results),
        }

        # 1. Core Retrieval Metrics (Recall, Precision, nDCG, Hit Rate)
        for k in k_values:
            top_k = results[:k]
            seen_rel: Set[str] = set()
            for item in top_k:
                matched = self._is_relevant(item, gold)
                if matched is not None:
                    seen_rel.add(matched)

            # Distinct relevant items only: counting positions let a duplicated
            # gold region inflate precision.
            hits = len(seen_rel)
            recall = min(1.0, len(seen_rel) / max(1, len(gold))) if gold else 0.0
            precision = hits / max(1, len(top_k)) if top_k else 0.0
            hit_rate = 1.0 if len(seen_rel) > 0 else 0.0
            ndcg = self._compute_ndcg(top_k, gold, k=k)

            metrics[f"recall_at_{k}"] = round(recall, 4)
            metrics[f"precision_at_{k}"] = round(precision, 4)
            metrics[f"hit_rate_at_{k}"] = round(hit_rate, 4)
            metrics[f"ndcg_at_{k}"] = round(ndcg, 4)

        # 2. Mean Reciprocal Rank (MRR) & MAP@10
        mrr = 0.0
        gold_ranks: List[int] = []
        for rank, item in enumerate(results, start=1):
            if self._is_relevant(item, gold) is not None:
                gold_ranks.append(rank)
                if mrr == 0.0:
                    mrr = 1.0 / rank

        metrics["mrr"] = round(mrr, 4)
        metrics["map_at_10"] = self._compute_map(results, gold, k=10)

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
            if self._is_relevant(item, gold) is not None
        )

        ccr = relevant_context_tokens / max(1, total_context_tokens) if total_context_tokens > 0 else 0.0
        noise_ratio = 1.0 - (metrics.get("precision_at_5", 0.0))
        metrics["context_compression_ratio"] = round(ccr, 4)
        metrics["noise_ratio"] = round(max(0.0, noise_ratio), 4)

        unique_doc_sources = len({str(item.get("document_id", "")) for item in results[:5] if item.get("document_id")})
        metrics["unique_evidence_ratio"] = (
            round(unique_doc_sources / max(1, min(5, len(results))), 4) if results else 0.0
        )

        # 5. Attribution & Groundedness (NLI Faithfulness & Hallucination Rate)
        answer_str = ""
        if answer:
            answer_str = answer.get("answer", "") if isinstance(answer, dict) else str(answer)
            attr_report = self.attributor.attribute(answer_str, evidence)
            faithfulness = attr_report.get("attribution_precision")
            # None means the check was disabled by config.  Emit nothing rather than a
            # value, so a switched-off gate cannot satisfy an SLA threshold.
            if faithfulness is not None:
                metrics["attribution_precision"] = round(float(faithfulness), 4)
                metrics["hallucination_rate"] = round(max(0.0, 1.0 - float(faithfulness)), 4)
            metrics["attribution_status"] = attr_report.get("status", "unknown")
            metrics["attribution_method"] = attr_report.get("method", "unknown")
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
            metrics["teds_score"] = table_metrics.get("teds_proxy_score", 0.0)
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

        if stratum:
            metrics["stratum"] = str(stratum)
        if latency_breakdown:
            metrics["latency_breakdown"] = latency_breakdown

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
                stratum=case.get("stratum"),
                latency_breakdown=case.get("latency_breakdown"),
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
            # A mean with no interval cannot distinguish a real gain from resampling
            # noise, which on a suite this size is the difference of one query.
            "confidence_intervals": self.confidence_intervals(results),
            "stage_latency_ms": self.stage_latency_summary(results),
            "per_stratum": self.stratified_summary(results),
            "target_verification": target_check,
            "detailed_results": results,
        }

    #: Metrics that get a bootstrap interval. Deliberately short: an interval per
    #: metric on all 22 would bury the four numbers any comparison actually turns on.
    CI_METRICS = ("recall_at_5", "ndcg_at_5", "mrr", "map_at_10", "attribution_precision", "latency_ms")

    def confidence_intervals(
        self,
        results: List[Dict[str, Any]],
        metrics: Optional[Iterable[str]] = None,
        confidence: float = 0.95,
        seed: int = 0,
        iterations: int = DEFAULT_ITERATIONS,
    ) -> Dict[str, Any]:
        """95% percentile-bootstrap interval per headline metric.

        *iterations* is a parameter because the caller knows what the numbers are for:
        a published report wants the default 10,000, while an ablation sweep computing
        intervals for eight systems and every stratum wants far fewer and says so in
        the artifact.
        """
        out: Dict[str, Any] = {}
        for name in metrics or self.CI_METRICS:
            values = [r[name] for r in results if isinstance(r.get(name), (int, float))]
            if not values:
                continue
            out[name] = bootstrap_ci(values, confidence=confidence, seed=seed, iterations=iterations).as_dict()
        return out

    @staticmethod
    def stage_latency_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Per-stage mean and p95 across the suite, from each case's breakdown.

        Answers "which stage regressed", which a single end-to-end number never could.
        Cases served from the response cache are excluded: their breakdown describes a
        cache lookup, and averaging those in would flatter every stage at once.
        """
        collected: Dict[str, List[float]] = {}
        counted = 0
        for record in results:
            breakdown = record.get("latency_breakdown") or {}
            stages = breakdown.get("stages_ms") or {}
            if not stages or breakdown.get("cached"):
                continue
            counted += 1
            for stage, elapsed in stages.items():
                collected.setdefault(stage, []).append(float(elapsed))

        if not counted:
            return {}

        summary: Dict[str, Any] = {"cases_with_breakdown": counted}
        for stage, values in collected.items():
            ordered = sorted(values)
            p95_index = max(0, int(round(0.95 * (len(ordered) - 1))))
            summary[stage] = {
                "mean_ms": round(sum(ordered) / len(ordered), 3),
                "p95_ms": round(ordered[p95_index], 3),
                "observations": len(ordered),
            }
        return summary

    def stratified_summary(
        self,
        results: List[Dict[str, Any]],
        iterations: int = DEFAULT_ITERATIONS,
    ) -> Dict[str, Any]:
        """Headline metrics per query stratum, with the interval for each.

        A corpus-wide mean can hold steady while tabular lookup collapses and prose
        improves; the strata are the only place that shows.
        """
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for record in results:
            stratum = record.get("stratum")
            if stratum:
                buckets.setdefault(str(stratum), []).append(record)

        out: Dict[str, Any] = {}
        for stratum, records in sorted(buckets.items()):
            latencies = [r.get("latency_ms", 0.0) for r in records]
            out[stratum] = {
                "n": len(records),
                "summary_metrics": self._aggregate_summary(records, latencies),
                "confidence_intervals": self.confidence_intervals(records, iterations=iterations),
            }
        return out

    def verify_target_achievements(self, summary: Dict[str, float], targets: Dict[str, float]) -> Dict[str, Any]:
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

    def _compute_ndcg(self, results: List[Dict[str, Any]], gold: Set[str], k: int) -> float:
        if not results or not gold:
            return 0.0

        dcg = 0.0
        seen_rel_ids: Set[str] = set()
        for rank, item in enumerate(results[:k], start=1):
            matched = self._is_relevant(item, gold)
            if matched is not None and matched not in seen_rel_ids:
                seen_rel_ids.add(matched)
                dcg += 1.0 / math.log2(rank + 1)

        ideal_hits = min(len(gold), k)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))

        score = dcg / idcg if idcg > 0 else 0.0
        return round(min(1.0, score), 4)

    def _compute_map(self, results: List[Dict[str, Any]], gold: Set[str], k: int = 10) -> float:
        if not results or not gold:
            return 0.0

        hits = 0
        sum_precisions = 0.0
        seen_rel: Set[str] = set()
        for rank, item in enumerate(results[:k], start=1):
            matched = self._is_relevant(item, gold)
            if matched is not None and matched not in seen_rel:
                seen_rel.add(matched)
                hits += 1
                sum_precisions += hits / rank

        return round(sum_precisions / max(1, len(gold)), 4)

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

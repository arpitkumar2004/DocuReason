#!/usr/bin/env python3
"""scripts/run_eval_suite.py - score the full system over the gold query set.

    python scripts/run_eval_suite.py --queries data/eval/queries.json

``run_ablations.py`` measures the *difference* between systems. This measures the system
itself: one pass over every gold query, with bootstrap intervals, a per-stratum
breakdown, the per-stage latency decomposition, and a census of what degraded.
``evaluate_system.py`` scores five hand-written cases against SLA targets over
``samples/`` - useful as a smoke test, but it is not an evaluation of a corpus.

Degradation is **counted**, not just logged. A suite whose retrieval quietly ran in
substring-scan mode scores like one that queried its indexes, and which of the two
happened is the whole question (audit T3).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docureason.tripath.evaluation.eval_harness import EvaluationHarness  # noqa: E402
from docureason.tripath.evaluation.gold_queries import load_eval_cases  # noqa: E402
from docureason.tripath.serving.query_service import QueryService  # noqa: E402
from docureason.tripath.utils import get_logger  # noqa: E402

logger = get_logger("run_eval_suite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", default="data/eval/queries.json", help="query set from build_eval_corpus.py")
    parser.add_argument("--filings", default="data/eval/filings", help="corpus input directory")
    parser.add_argument("--corpus", default="data/eval/corpus", help="ingested corpus / index directory")
    parser.add_argument("--out", default="artifacts/evaluation", help="where to write the report")
    parser.add_argument("--limit", type=int, default=0, help="use only N queries (0 = all)")
    parser.add_argument("--stratified", action="store_true", help="sample --limit evenly across strata")
    parser.add_argument("--iterations", type=int, default=2000, help="bootstrap resamples (default: 2000)")
    parser.add_argument("--run-name", default="eval_suite", help="report file stem")
    return parser.parse_args()


def run_suite(
    service: QueryService,
    harness: EvaluationHarness,
    cases: List[Dict[str, Any]],
    iterations: int,
) -> Dict[str, Any]:
    """Execute every case against *service* and score it through *harness*."""
    scored: List[Dict[str, Any]] = []
    degradation: Counter = Counter()
    retrieval_modes: Counter = Counter()
    failures: List[Dict[str, str]] = []
    cached_hits = 0

    started = time.perf_counter()
    for position, case in enumerate(cases, start=1):
        query = str(case.get("query", ""))
        query_started = time.perf_counter()
        try:
            response = service.query(query)
        except Exception as exc:  # one bad query must not void the suite
            logger.error("Query %d/%d failed: %s", position, len(cases), exc)
            failures.append({"query": query, "error": repr(exc)})
            continue
        latency_ms = (time.perf_counter() - query_started) * 1000.0

        for label in response.get("degradation") or []:
            degradation[str(label)] += 1
        # The per-path mode lives under retrieval_stats["modality_modes"], each value the
        # path's own stats dict. index_backed vs substring_scan is the difference between
        # querying the indexes the framework builds and scanning strings past them.
        for modality, stats in ((response.get("retrieval_stats") or {}).get("modality_modes") or {}).items():
            mode = stats.get("mode") if isinstance(stats, dict) else stats
            if mode:
                retrieval_modes[f"{modality}:{mode}"] += 1
        cached_hits += 1 if response.get("cached") else 0

        scored.append(
            harness.evaluate_single(
                query=query,
                results=response.get("results", []),
                relevant_ids=case.get("relevant_ids", []),
                answer=response.get("answer"),
                ground_truth_answer=case.get("ground_truth_answer"),
                evidence=response.get("retrieved_evidence"),
                sql_executed=(response.get("sql_execution") or {}).get("executed"),
                latency_ms=round(latency_ms, 3),
                stratum=case.get("stratum"),
                latency_breakdown=response.get("latency_breakdown"),
            )
        )
        if position % 10 == 0:
            logger.info("Scored %d/%d queries", position, len(cases))

    latencies = [record.get("latency_ms", 0.0) for record in scored]
    return {
        "queries_requested": len(cases),
        "queries_scored": len(scored),
        "failures": failures,
        "wall_clock_sec": round(time.perf_counter() - started, 2),
        "summary_metrics": harness._aggregate_summary(scored, latencies),
        "confidence_intervals": harness.confidence_intervals(scored, iterations=iterations),
        "stage_latency_ms": harness.stage_latency_summary(scored),
        "per_stratum": harness.stratified_summary(scored, iterations=iterations),
        # Counted, not logged: a substring-scan run and an index-backed run score alike.
        "degradation_counts": dict(degradation.most_common()),
        "retrieval_modes": dict(retrieval_modes.most_common()),
        "cache_hits": cached_hits,
        "per_query": scored,
    }


#: Summary key -> the per-case metric whose bootstrap interval belongs beside it. The
#: harness aggregates under ``average_*`` but computes intervals under the raw per-case
#: name, so a naive lookup silently prints every interval as blank.
_CI_ALIASES = {
    "average_recall_at_5": "recall_at_5",
    "average_ndcg_at_5": "ndcg_at_5",
    "mean_latency_ms": "latency_ms",
}


def _cell(metrics: Dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    return "n/a" if not isinstance(value, (int, float)) else f"{value:.4f}"


def print_report(report: Dict[str, Any]) -> None:
    summary = report["summary_metrics"]
    intervals = report.get("confidence_intervals") or {}

    print("")
    print(f"Scored {report['queries_scored']}/{report['queries_requested']} queries in {report['wall_clock_sec']}s")
    if report["failures"]:
        print(f"  {len(report['failures'])} query(ies) failed - listed under 'failures' in the report")
    print("")
    print(f"  {'metric':30s} {'value':>12s}   95% CI")
    for metric in sorted(summary):
        value = summary[metric]
        if not isinstance(value, (int, float)):
            continue
        interval = intervals.get(_CI_ALIASES.get(metric, metric)) or {}
        low, high = interval.get("ci_low"), interval.get("ci_high")
        span = "" if low is None or high is None else f"[{low:.4f}, {high:.4f}]"
        print(f"  {metric:30s} {value:12.4f}   {span}")

    per_stratum = report.get("per_stratum") or {}
    if per_stratum:
        print("")
        print("  per stratum - a corpus-wide mean can hold while one stratum collapses:")
        for stratum in sorted(per_stratum):
            block = per_stratum[stratum] or {}
            metrics = block.get("summary_metrics") or {}
            print(
                f"    {stratum:18s} n={block.get('n', '?'):<5} "
                f"recall@5={_cell(metrics, 'average_recall_at_5')}  "
                f"ndcg@5={_cell(metrics, 'average_ndcg_at_5')}  "
                f"token_f1={_cell(metrics, 'token_f1')}"
            )

    stages = report.get("stage_latency_ms") or {}
    rows = [
        (stage, block["mean_ms"], block.get("p95_ms"))
        for stage, block in stages.items()
        if isinstance(block, dict) and isinstance(block.get("mean_ms"), (int, float))
    ]
    if rows:
        print("")
        print(f"  per-stage latency over {stages.get('cases_with_breakdown', '?')} uncached case(s), ms:")
        for stage, mean_ms, p95_ms in sorted(rows, key=lambda row: -row[1]):
            print(f"    {stage:20s} mean={mean_ms:10.2f}  p95={p95_ms if p95_ms is None else round(p95_ms, 2)}")

    if report.get("retrieval_modes"):
        print("")
        print("  retrieval mode (index_backed vs substring_scan is the whole question):")
        for label, count in report["retrieval_modes"].items():
            print(f"    {label:44s} {count}")

    print("")
    if report.get("degradation_counts"):
        print("  degradation (label: queries affected):")
        for label, count in report["degradation_counts"].items():
            print(f"    {label:44s} {count}")
    else:
        print("  degradation: none reported on any query")


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        cases = load_eval_cases(args.queries, limit=args.limit, stratified=args.stratified)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    logger.info("Loaded %d evaluation case(s) from %s", len(cases), args.queries)

    service = QueryService(input_dir=args.filings, output_dir=args.corpus, verbose=False)
    harness = EvaluationHarness(output_dir=out)

    report = run_suite(service, harness, cases, iterations=args.iterations)
    report["corpus"] = {"filings": str(args.filings), "corpus": str(args.corpus), "queries": str(args.queries)}
    report["query_subset"] = {
        "requested_limit": args.limit,
        "stratified_sample": bool(args.stratified),
        "is_full_set": args.limit == 0,
    }
    report["bootstrap_iterations"] = args.iterations

    path = out / f"{args.run_name}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_report(report)
    print("")
    print(f"report: {path}")
    return 0 if report["queries_scored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

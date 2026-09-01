#!/usr/bin/env python3
"""Run the seven-row ablation matrix, and the naive-RAG baseline, for real.

    python scripts/run_ablations.py --queries data/eval/queries.json --limit 20

Every ablation re-runs **every** query with one component disabled, scores both runs
through the same harness, and reports the difference as a paired comparison with a
bootstrap interval and a sign-flip p-value.

Cost: (1 + ablations) x queries full pipeline executions. Seven ablations over 200
queries is 1,600 executions, which on CPU is hours - so ``--limit`` exists, and a run
that used it says so in the artifact rather than presenting a subset as the matrix.

``Documentations/eval_method.md`` §21 publishes a delta for each row. Those numbers
predate any runner that could produce them; the report prints measured against claimed
so an unsupported claim is visible instead of inherited.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docureason.tripath.evaluation.ablation import (  # noqa: E402
    ABLATIONS,
    ABLATIONS_BY_KEY,
    AblationRunner,
    markdown_matrix,
)
from docureason.tripath.evaluation.baselines import NaiveRAGBaseline  # noqa: E402
from docureason.tripath.evaluation.eval_harness import EvaluationHarness  # noqa: E402
from docureason.tripath.evaluation.gold_queries import load_eval_cases  # noqa: E402
from docureason.tripath.serving.query_service import QueryService  # noqa: E402
from docureason.tripath.utils import get_logger  # noqa: E402

logger = get_logger("run_ablations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", default="data/eval/queries.json", help="query set from build_eval_corpus.py")
    parser.add_argument("--filings", default="data/eval/filings", help="corpus input directory")
    parser.add_argument("--corpus", default="data/eval/corpus", help="ingested corpus / index directory")
    parser.add_argument("--out", default="artifacts/ablations", help="where to write the report")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N queries (0 = all)")
    parser.add_argument("--stratified", action="store_true", help="sample --limit evenly across strata")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=2000, help="bootstrap resamples (default: 2000)")
    parser.add_argument("--only", nargs="*", default=None, help="run only these ablation keys")
    parser.add_argument(
        "--ingestion-corpus",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help=(
            "corpus for an ingestion-scoped ablation, repeatable, e.g. "
            "--ingestion-corpus no_layout_parser=data/eval/corpus_no_layout. Without one, that "
            "row is reported as not executed: the component runs once at ingestion, so it "
            "cannot be switched off per query. Build the corpus with build_eval_corpus.py "
            "--corpus-dir <path> --no-layout-parser --skip-derive"
        ),
    )
    parser.add_argument("--no-baseline", action="store_true", help="skip the naive-RAG comparison")
    return parser.parse_args()


def load_cases(path: Path, limit: int, stratified: bool) -> List[Dict[str, Any]]:
    """Thin CLI wrapper: the loader is shared with scripts/run_eval_suite.py so the two
    commands cannot end up sampling different query sets under the same corpus name."""
    try:
        return load_eval_cases(path, limit=limit, stratified=stratified)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def matrix_document(report: Dict[str, Any], matrix: str) -> str:
    """The matrix table, plus the provenance a reader needs in order to weigh it.

    Which query set, how many queries, and how many rows actually executed. Without those
    three facts a three-query subset over a fixture corpus renders identically to the full
    matrix over real filings - and the table is the part that gets quoted.
    """
    subset = report.get("query_subset") or {}
    rows = report.get("ablations") or []
    unexecuted = report.get("not_executed") or []
    sample_note = " (stratified sample)" if subset.get("stratified_sample") else ""

    document = ["# Measured ablation matrix", ""]
    if not subset.get("is_full_matrix"):
        document += ["**Subset run - this is not the full matrix.**", ""]
    document += [
        f"- queries: **{report.get('queries')}** from `{subset.get('source')}`{sample_note}",
        f"- rows executed: **{report.get('executed')} of {len(rows)}**"
        + ("" if not unexecuted else ", not executed: " + ", ".join(unexecuted)),
        f"- bootstrap iterations: {report.get('bootstrap_iterations')}, seed {report.get('seed')}",
        f"- ingestion-scoped corpora: {report.get('ingestion_corpora') or 'none supplied'}",
        "",
        matrix,
        "",
    ]
    return "\n".join(document)


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = load_cases(Path(args.queries), args.limit, args.stratified)
    logger.info("Loaded %d evaluation case(s)", len(cases))

    # Validated before anything expensive is constructed: loading the corpus and building
    # the baseline index takes minutes, and a typo in a path is not worth waiting for.
    ingestion_corpora: Dict[str, Path] = {}
    for spec in args.ingestion_corpus:
        key, separator, raw_path = spec.partition("=")
        key, raw_path = key.strip(), raw_path.strip()
        if not separator or not key or not raw_path:
            raise SystemExit(f"--ingestion-corpus expects KEY=PATH, got {spec!r}")
        if key not in ABLATIONS_BY_KEY:
            raise SystemExit(f"unknown ablation key {key!r}; known keys: {sorted(ABLATIONS_BY_KEY)}")
        corpus_path = Path(raw_path)
        if not (corpus_path / "corpus.json").exists():
            raise SystemExit(f"no corpus.json under {corpus_path} for ablation {key!r}")
        ingestion_corpora[key] = corpus_path

    service = QueryService(input_dir=args.filings, output_dir=args.corpus, verbose=False)
    harness = EvaluationHarness(output_dir=out)

    ablations = list(ABLATIONS)
    if args.only:
        wanted = set(args.only)
        ablations = [ablation for ablation in ablations if ablation.key in wanted]
        if not ablations:
            raise SystemExit(f"no ablations matched {sorted(wanted)}")

    baseline = None
    if not args.no_baseline:
        baseline = NaiveRAGBaseline(config=service.config, generator=service.generator)
        stats = baseline.build(service.documents)
        logger.info("Naive baseline: %s", stats)

    # Ingestion-scoped rows answer from a second corpus, built with the component off. Its
    # gold ids are the un-ablated corpus's, so the runner reports the retrieval metrics as
    # not comparable and carries the row on answer quality plus the corpus census.
    ingestion_services: Dict[str, Any] = {}
    for key, corpus_path in ingestion_corpora.items():
        logger.info("Ablation %s will answer from corpus %s", key, corpus_path)
        ingestion_services[key] = QueryService(input_dir=args.filings, output_dir=str(corpus_path), verbose=False)

    runner = AblationRunner(
        service=service,
        harness=harness,
        ablations=ablations,
        iterations=args.iterations,
        seed=args.seed,
        ingestion_services=ingestion_services,
    )
    report = runner.run(cases, baseline=baseline)
    report["query_subset"] = {
        "requested_limit": args.limit,
        "stratified_sample": bool(args.stratified),
        "queries_used": len(cases),
        "source": str(args.queries),
        # Stated in the artifact because a subset run is not the matrix, and a reader
        # of the JSON has no other way to tell the two apart. An unexecuted row
        # disqualifies the run too: six rows of seven is not the matrix either.
        "is_full_matrix": args.limit == 0 and not args.only and not report["not_executed"],
    }
    report["ingestion_corpora"] = {key: str(path) for key, path in ingestion_corpora.items()}

    report_path = out / "ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    matrix = markdown_matrix(report)
    matrix_path = out / "ablation_matrix.md"
    matrix_path.write_text(matrix_document(report, matrix), encoding="utf-8")

    print("")
    print(matrix)
    print("")
    if report.get("baseline"):
        print("Full system vs naive RAG baseline:")
        for metric, comparison in report["baseline"]["full_vs_baseline"].items():
            delta = comparison.get("delta")
            low, high = comparison.get("delta_ci_low"), comparison.get("delta_ci_high")
            p_value = comparison.get("p_value")
            if delta is None:
                continue
            print(
                f"  {metric:24s} {delta:+.4f}  95% CI [{low:+.4f}, {high:+.4f}]  "
                f"p={p_value if p_value is None else round(p_value, 4)}"
            )
    print("")
    print(f"not executed: {report['not_executed'] or 'none'}")
    print(f"report: {report_path}")
    print(f"matrix: {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

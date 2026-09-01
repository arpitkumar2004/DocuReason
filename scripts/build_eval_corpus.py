#!/usr/bin/env python3
"""Build the evaluation corpus: fetch real filings, ingest them, derive gold queries.

    export DOCUREASON_SEC_USER_AGENT="Your Org corpus-builder (you@example.com)"
    python scripts/build_eval_corpus.py --filings 24 --per-stratum 60

Four steps, each idempotent, so a failure part-way can be resumed:

1. **fetch**    SEC EDGAR 10-K primary documents -> ``data/eval/filings`` (cached on disk)
2. **ingest**   the full DocuReason pipeline      -> ``data/eval/corpus``
3. **derive**   stratified queries with gold region ids -> ``data/eval/queries.json``
4. **validate** every gold id resolves; every phrase anchor is still unique

Step 2 is the expensive one: layout parsing a 10-K on CPU takes minutes per filing, and
gold region ids only exist after it, so the query set cannot be built without it.

The output is reproducible from ``provenance.json`` (filing URLs plus SHA-256) and the
generator seed. That is the difference between "we evaluated on 24 filings" as a claim
and as a checkable statement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docureason.pipeline import DocuReasonPipeline  # noqa: E402
from docureason.tripath.config import DocuReasonConfig  # noqa: E402
from docureason.tripath.evaluation.edgar import (  # noqa: E402
    DEFAULT_COMPANIES,
    EdgarAccessError,
    fetch_filings,
)
from docureason.tripath.evaluation.gold_queries import generate_queries, validate_queries  # noqa: E402
from docureason.tripath.utils import get_logger  # noqa: E402

logger = get_logger("build_eval_corpus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/eval", help="output root (default: data/eval)")
    parser.add_argument("--filings", type=int, default=24, help="how many filings to fetch (default: 24)")
    parser.add_argument("--form", default="10-K", help="EDGAR form type (default: 10-K)")
    parser.add_argument("--per-company", type=int, default=2, help="filings per company (default: 2)")
    parser.add_argument("--per-stratum", type=int, default=60, help="target queries per stratum (default: 60)")
    parser.add_argument("--seed", type=int, default=0, help="generator seed (default: 0)")
    parser.add_argument("--skip-fetch", action="store_true", help="reuse whatever is already in filings/")
    parser.add_argument("--skip-ingest", action="store_true", help="reuse an existing corpus.json")
    parser.add_argument("--corpus-dir", default=None, help="ingest into this directory (default: <out>/corpus)")
    parser.add_argument(
        "--no-layout-parser",
        action="store_true",
        help="ingest with ingestion.enable_layout_parser=false: the corpus the no_layout_parser ablation row needs",
    )
    parser.add_argument(
        "--skip-derive",
        action="store_true",
        help="build the corpus only, leaving queries.json alone (use with --no-layout-parser)",
    )
    return parser.parse_args()


def step_fetch(out: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    filings_dir = out / "filings"
    if args.skip_fetch:
        existing = sorted(p for p in filings_dir.glob("*.htm*"))
        logger.info("Skipping fetch; %d filing(s) already on disk", len(existing))
        return [{"path": str(p)} for p in existing]

    try:
        fetched = fetch_filings(
            filings_dir,
            companies=DEFAULT_COMPANIES,
            form=args.form,
            per_company=args.per_company,
            limit=args.filings,
        )
    except EdgarAccessError as exc:
        raise SystemExit(f"EDGAR access failed: {exc}") from exc

    if not fetched:
        raise SystemExit("no filings fetched; refusing to build a corpus from nothing")
    return [item.as_dict() for item in fetched]


def step_ingest(out: Path, args: argparse.Namespace) -> Dict[str, Any]:
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else out / "corpus"
    corpus_path = corpus_dir / "corpus.json"

    if args.skip_ingest and corpus_path.exists():
        logger.info("Skipping ingest; reusing %s", corpus_path)
        return json.loads(corpus_path.read_text(encoding="utf-8"))

    config = None
    if args.no_layout_parser:
        # The corpus the no_layout_parser ablation row answers from. It goes in its own
        # directory: the gold ids belong to the layout-parsed corpus, and overwriting that
        # corpus would relabel every one of them. This is a second observation, not a
        # replacement.
        config = DocuReasonConfig.load_from_yaml("configs/config.yaml")
        config.ingestion.enable_layout_parser = False
        logger.warning("Layout parser disabled; ingesting the ablation corpus into %s", corpus_dir)

    started = time.perf_counter()
    report = DocuReasonPipeline(
        input_dir=out / "filings", output_dir=corpus_dir, config=config, verbose=False
    ).run()
    logger.info(
        "Ingested %s document(s) / %s chunk(s) in %.1f min",
        report.get("document_count"),
        report.get("chunk_count"),
        (time.perf_counter() - started) / 60.0,
    )
    if report.get("degradation"):
        logger.warning("Ingestion degradation: %s", report["degradation"])
    return json.loads(corpus_path.read_text(encoding="utf-8"))


def step_derive(out: Path, corpus: Dict[str, Any], filings: List[Dict[str, Any]], args: argparse.Namespace) -> Path:
    report = generate_queries(corpus, per_stratum=args.per_stratum, seed=args.seed)
    records = [query.as_dict() for query in report.queries]
    validation = validate_queries(records, corpus)

    payload = {
        "dataset": "docureason-edgar-eval",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "form": args.form,
        "filing_count": len(filings),
        "provenance": "Gold regions are derived programmatically from the ingested corpus, "
        "not human-annotated: table cells give a known row/column intersection, and prose "
        "and figure anchors are phrases verified unique across the corpus. Derivable queries "
        "over-represent lookup and cannot express multi-hop reasoning.",
        "generation_report": report.as_dict(),
        "validation": validation,
        "queries": records,
    }

    path = out / "queries.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %d query(ies) to %s", len(records), path)

    if not validation["valid"]:
        logger.error(
            "Validation failed: %d dangling gold id(s), %d non-unique anchor(s)",
            len(validation["dangling_gold_ids"]),
            len(validation["non_unique_anchors"]),
        )
    return path


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    filings = step_fetch(out, args)
    corpus = step_ingest(out, args)

    if args.skip_derive:
        # Deriving here would relabel the gold ids against whatever corpus this run built.
        corpus_dir = Path(args.corpus_dir) if args.corpus_dir else out / "corpus"
        print("")
        print("Corpus built; query derivation skipped")
        print(f"  filings         : {len(filings)}")
        print(f"  documents       : {len(corpus.get('documents', []))}")
        print(f"  layout parser   : {'off' if args.no_layout_parser else 'on'}")
        print(f"  written to      : {corpus_dir}")
        return 0

    path = step_derive(out, corpus, filings, args)

    payload = json.loads(path.read_text(encoding="utf-8"))
    report = payload["generation_report"]
    validation = payload["validation"]

    print("")
    print("Evaluation corpus")
    print(f"  filings         : {len(filings)}")
    print(f"  documents       : {report['documents']}")
    print(f"  regions scanned : {report['regions_scanned']}")
    print(f"  queries         : {report['query_count']}")
    for stratum, count in sorted(report["per_stratum"].items()):
        print(f"      {stratum:16s}: {count}")
    print(f"  rejected        : {report['rejected']}")
    print(f"  validation      : {'PASS' if validation['valid'] else 'FAIL'}")
    print(f"  written to      : {path}")
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""scripts/plot_results.py - figures and a derived-insight digest from the reports.

    python scripts/plot_results.py

Reads ``artifacts/evaluation/eval_suite.json`` (from run_eval_suite.py) and
``artifacts/ablations/ablation_report.json`` (from run_ablations.py) and writes
``artifacts/figures/*.png`` + ``*.pdf`` plus ``INSIGHTS.md``.

Either report may be absent: the figures it would have produced are skipped and listed in
the digest, because an axis with nothing on it reads as "measured, and flat".

Needs the plotting extra, which the framework itself does not depend on:

    pip install 'docureason-framework[figures]'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docureason.tripath.evaluation.figures import DEFAULT_FORMATS, build_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-report", default="artifacts/evaluation/eval_suite.json")
    parser.add_argument("--ablation-report", default="artifacts/ablations/ablation_report.json")
    parser.add_argument("--out", default="artifacts/figures", help="where to write figures and INSIGHTS.md")
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="comma-separated output formats (default: png,pdf - raster for notebooks, vector for papers)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    formats = tuple(part.strip() for part in args.formats.split(",") if part.strip())

    result = build_all(
        eval_report=args.eval_report,
        ablation_report=args.ablation_report,
        out_dir=args.out,
        formats=formats,
    )

    print("")
    for path in result["figures"]:
        print(f"  wrote {path}")
    for item in result["skipped"]:
        print(f"  skipped {item}")
    print("")
    print(f"insights: {result['insights']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

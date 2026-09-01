#!/usr/bin/env python3
"""Metric regression gate: fail when harness metrics move on a frozen fixture.

    python scripts/check_metric_regression.py                # compare, exit 1 on drift
    python scripts/check_metric_regression.py --tolerance 0  # exact match
    python scripts/check_metric_regression.py --update       # re-record the baseline

359 lines of metric arithmetic - nDCG, MAP, MRR, compression, noise, attribution -
had no test that would notice if any of it changed. A refactor could shift every
reported number and CI would stay green, which is how a published benchmark table comes
to describe a system that no longer exists.

The fixture is **frozen retrieval output**, not a live pipeline run: fixed result lists
with fixed gold ids, no models, no network, no corpus. That is deliberate. This gate
answers exactly one question - "did the metric definitions change?" - and a fixture
that re-ran retrieval would answer it only when the embedding model, the reranker and
the corpus all happened to hold still.

Updating the baseline is a deliberate act with a diff to review. If a change moves
these numbers, the pull request should say why.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docureason.tripath.evaluation.eval_harness import EvaluationHarness  # noqa: E402

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "metric_regression_fixture.json"
BASELINE_PATH = ROOT / "tests" / "fixtures" / "metric_regression_baseline.json"

#: Default drift allowed per metric. 0.01 is one percentage point: tight enough to
#: catch a changed formula, loose enough to survive floating-point reassociation.
DEFAULT_TOLERANCE = 0.01


def load_fixture(path: Path = FIXTURE_PATH) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["cases"]


def compute(cases: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    """Score the fixture and return only the deterministic parts of the report.

    Latency and throughput are dropped: they measure the machine, and gating them
    would make the build fail on a busy runner.
    """
    harness = EvaluationHarness(output_dir=output_dir)
    report = harness.evaluate_suite(cases)

    summary = {
        key: value
        for key, value in (report.get("summary_metrics") or {}).items()
        if not _is_machine_dependent(key) and isinstance(value, (int, float))
    }
    strata = {
        name: {
            key: value
            for key, value in (bucket.get("summary_metrics") or {}).items()
            if not _is_machine_dependent(key) and isinstance(value, (int, float))
        }
        for name, bucket in (report.get("per_stratum") or {}).items()
    }
    return {
        "total_queries": report.get("total_queries"),
        "summary_metrics": summary,
        "per_stratum": strata,
    }


def _is_machine_dependent(metric: str) -> bool:
    return any(token in metric for token in ("latency", "qps", "duration", "per_second"))


def flatten(payload: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten the nested report into ``dotted.path -> value`` for comparison."""
    flat: Dict[str, float] = {}
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{path}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[path] = float(value)
    return flat


def compare(current: Dict[str, Any], baseline: Dict[str, Any], tolerance: float) -> Tuple[bool, List[str]]:
    """Return ``(ok, findings)``. A missing or added metric is drift too."""
    now = flatten(current)
    before = flatten(baseline)
    findings: List[str] = []

    for name in sorted(set(before) - set(now)):
        findings.append(f"REMOVED  {name}: was {before[name]:.4f}, no longer reported")
    for name in sorted(set(now) - set(before)):
        findings.append(f"NEW      {name}: now {now[name]:.4f}, not in the baseline")

    for name in sorted(set(now) & set(before)):
        delta = now[name] - before[name]
        if abs(delta) > tolerance:
            findings.append(f"MOVED    {name}: {before[name]:.4f} -> {now[name]:.4f} ({delta:+.4f})")

    return not findings, findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--update", action="store_true", help="overwrite the baseline with current values")
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    args = parser.parse_args()

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        current = compute(load_fixture(args.fixture), Path(tmp))

    if args.update:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Baseline rewritten: {args.baseline} ({len(flatten(current))} metrics)")
        return 0

    if not args.baseline.exists():
        print(f"No baseline at {args.baseline}. Record one with --update.")
        return 1

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    ok, findings = compare(current, baseline, args.tolerance)

    if ok:
        print(f"Metric regression gate: PASS ({len(flatten(current))} metrics within +/-{args.tolerance})")
        return 0

    print(f"Metric regression gate: FAIL ({len(findings)} finding(s), tolerance +/-{args.tolerance})")
    for finding in findings:
        print(f"  {finding}")
    print("\nIf the change is intended, re-record with:")
    print("  python scripts/check_metric_regression.py --update")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

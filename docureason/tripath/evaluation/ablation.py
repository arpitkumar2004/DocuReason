"""ablation.py — measure what each architectural block actually contributes.

What was here before: a 24-line function that subtracted one pre-computed
``average_recall_at_5`` from another, with no caller anywhere in the repository.
Meanwhile ``Documentations/eval_method.md`` §21 published a seven-row ablation matrix
with deltas to one decimal place (-42.5%, -61.2%, -21.4%, ...) that nothing in the
codebase could have produced.  Three of those components could not even be switched
off: there was no flag for the sparse path, the cross-encoder, the heading penalty or
the SQL retriever, so "remove X and re-measure" was not an operation the system
supported.

This runner executes the matrix.  For each ablation it re-runs **every query** with
that component disabled, scores both runs through the same harness, and reports the
delta as a *paired* comparison: mean difference, bootstrap confidence interval and a
sign-flip p-value.  Paired because query-to-query variance dwarfs the between-system
variance - the same query is hard for both systems, and only the paired difference
cancels that out.

Two honesty mechanisms are built into the report:

* ``documented_claim`` carries each row's published delta next to the measured one, so
  a claim the measurement does not support is visible in the artifact rather than
  surviving in a table nobody re-derived.
* an ablation that could not be executed is reported as ``executed: false`` with a
  reason.  It is never silently dropped, and it never appears as a delta of zero -
  which is what "not measured" would otherwise look like.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from docureason.tripath.evaluation.statistics import DEFAULT_ITERATIONS, compare_paired
from docureason.tripath.utils import get_logger

logger = get_logger(__name__)

#: Metrics compared for every ablation. Retrieval quality, ranking quality, ordering,
#: faithfulness and latency - enough to distinguish "found less" from "ranked worse".
COMPARED_METRICS = (
    "recall_at_5",
    "precision_at_5",
    "ndcg_at_5",
    "mrr",
    "map_at_10",
    "hit_rate_at_5",
    "attribution_precision",
    "hallucination_rate",
    "duplicate_rank_ratio",
    # Answer quality: the parent-expansion row's primary metric is token_f1, so leaving
    # it out of this tuple rendered that row as "n/a" - the comparison was never
    # computed for the one metric its own claim is about.
    "token_f1",
    "exact_match",
    "latency_ms",
)

#: Metrics scored against gold **region ids**. Their values only mean something inside
#: the corpus those ids were derived from. An ingestion-scoped ablation rebuilds the
#: corpus, so the same sentence lives in a region with a different id and every metric
#: below collapses toward zero whether retrieval degraded or not. For such a row they
#: are reported as not comparable rather than as a delta: a -100% recall that actually
#: measures the label space would be the most misleading number in the artifact.
CORPUS_BOUND_METRICS = frozenset(
    {
        "recall_at_5",
        "precision_at_5",
        "ndcg_at_5",
        "mrr",
        "map_at_10",
        "hit_rate_at_5",
    }
)

#: Printed in place of a corpus-bound delta on a cross-corpus row.
NOT_COMPARABLE_REASON = (
    "scored against gold region ids derived from the un-ablated corpus; this row ran a "
    "corpus re-ingested with the component disabled, so the label space differs and the "
    "delta would measure region identity, not retrieval"
)

#: What a cross-corpus row reports instead. Scored against the ground-truth answer text,
#: which a corpus rebuild does not change, so it stays a paired measurement.
ANSWER_LEVEL_METRIC = "token_f1"


@dataclass(frozen=True)
class Ablation:
    """One row of the ablation matrix."""

    key: str
    label: str
    #: Per-query config overrides that disable the component. Empty for an ablation
    #: that can only be applied at ingestion time.
    overrides: Dict[str, Any]
    #: ``"query"`` (a per-query override suffices) or ``"ingestion"`` (needs a corpus
    #: rebuilt with the component off - the layout parser runs once, not per query).
    scope: str
    #: The metric the documented matrix claims this component moves.
    primary_metric: str
    #: The published delta from Documentations/eval_method.md §21, as a percentage.
    documented_claim_pct: Optional[float]
    rationale: str


#: The seven ablations from Documentations/eval_method.md §21, in that order.
ABLATIONS: Tuple[Ablation, ...] = (
    Ablation(
        key="no_layout_parser",
        label="Remove Docling layout parser",
        overrides={},
        scope="ingestion",
        primary_metric="recall_at_5",
        documented_claim_pct=-42.5,
        rationale="Layout-aware parsing is claimed to be essential for multi-column and tabular PDFs.",
    ),
    Ablation(
        key="no_table_sql",
        label="Remove TableSQL retriever (DuckDB)",
        overrides={"enable_table_sql": False},
        scope="query",
        primary_metric="recall_at_5",
        documented_claim_pct=-61.2,
        rationale="Dense vectors are claimed to fail at exact numerical arithmetic.",
    ),
    Ablation(
        key="no_cross_encoder",
        label="Remove cross-encoder reranker",
        overrides={"enable_cross_encoder": False},
        scope="query",
        primary_metric="ndcg_at_5",
        documented_claim_pct=-21.4,
        rationale="RRF rank output is claimed to need joint self-attention rescoring.",
    ),
    Ablation(
        key="no_parent_expansion",
        label="Remove parent-child expansion",
        overrides={"enable_parent_expansion": False},
        scope="query",
        primary_metric="token_f1",
        documented_claim_pct=-18.6,
        rationale="Isolated child chunks are claimed to lack context for QA.",
    ),
    Ablation(
        key="no_heading_penalty",
        label="Remove short-heading penalty",
        overrides={"enable_heading_penalty": False},
        scope="query",
        primary_metric="ndcg_at_5",
        documented_claim_pct=-14.2,
        rationale="Heading-only chunks are claimed to crowd out real content without the penalty.",
    ),
    Ablation(
        key="no_sparse_path",
        label="Remove BM25S sparse path",
        overrides={"enable_sparse_retrieval": False},
        scope="query",
        primary_metric="recall_at_5",
        documented_claim_pct=-28.9,
        rationale="Dense search is claimed to miss exact jargon and identifiers.",
    ),
    Ablation(
        key="no_claim_verification",
        label="Remove claim-support verification",
        overrides={"enable_nli": False},
        scope="query",
        primary_metric="hallucination_rate",
        documented_claim_pct=24.8,
        rationale="Claim checking is claimed to be vital against hallucination.",
    ),
)

ABLATIONS_BY_KEY: Dict[str, Ablation] = {ablation.key: ablation for ablation in ABLATIONS}


@dataclass
class SystemRun:
    """Every per-query metric vector for one system configuration."""

    name: str
    #: metric name -> value per query, in query order (None where unmeasured).
    metrics: Dict[str, List[Optional[float]]] = field(default_factory=dict)
    #: The harness's own suite report for this configuration.
    report: Dict[str, Any] = field(default_factory=dict)
    failures: int = 0

    def values(self, metric: str) -> List[Optional[float]]:
        return self.metrics.get(metric, [])


class AblationRunner:
    """Execute the ablation matrix against a live query service and an eval harness."""

    def __init__(
        self,
        service: Any,
        harness: Any,
        ablations: Sequence[Ablation] = ABLATIONS,
        compared_metrics: Sequence[str] = COMPARED_METRICS,
        iterations: int = DEFAULT_ITERATIONS,
        seed: int = 0,
        ingestion_services: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        service
            A built ``QueryService`` (or anything with ``query(text, overrides=...)``).
        harness
            An ``EvaluationHarness``; the *same* one scores every configuration, which
            is what makes the runs comparable.
        ingestion_services
            Optional ``{ablation_key: service}`` for ingestion-scoped ablations, each
            built from a corpus ingested with that component off. Absent keys are
            reported as not executed rather than skipped silently.
        """
        self.service = service
        self.harness = harness
        self.ablations = list(ablations)
        self.compared_metrics = list(compared_metrics)
        self.iterations = iterations
        self.seed = seed
        self.ingestion_services = dict(ingestion_services or {})

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, cases: Sequence[Dict[str, Any]], baseline: Optional[Any] = None) -> Dict[str, Any]:
        """Run the full system, every ablation, and optionally a baseline system.

        *cases* are ``{"query", "relevant_ids", ...}`` dicts - the same shape the
        harness suite takes. *baseline* is anything exposing ``query(text)``, scored
        through the identical harness so the comparison is not confounded by scoring.
        """
        if not cases:
            raise ValueError("no evaluation cases: an ablation over zero queries measures nothing")

        started = time.perf_counter()
        full = self._run_system("full", cases, overrides=None, service=self.service)

        rows: List[Dict[str, Any]] = []
        systems: Dict[str, Any] = {"full": full.report.get("summary_metrics", {})}

        for ablation in self.ablations:
            service = self._service_for(ablation)
            if service is None:
                rows.append(self._not_executed(ablation))
                logger.warning("Ablation %s not executed: %s", ablation.key, rows[-1]["reason"])
                continue

            overrides = ablation.overrides or None
            run = self._run_system(ablation.key, cases, overrides=overrides, service=service)
            systems[ablation.key] = run.report.get("summary_metrics", {})
            # An ingestion-scoped row runs a different service over a different corpus,
            # so metrics keyed on gold region ids stop being paired measurements of the
            # same thing. Decided from the service identity rather than from scope alone,
            # because scope says what the ablation is, not which corpus answered.
            rows.append(self._compare(ablation, full, run, cross_corpus=service is not self.service))

        report: Dict[str, Any] = {
            "queries": len(cases),
            "seed": self.seed,
            "bootstrap_iterations": self.iterations,
            "compared_metrics": list(self.compared_metrics),
            "systems": systems,
            "ablations": rows,
            "wall_clock_sec": round(time.perf_counter() - started, 2),
            "executed": sum(1 for row in rows if row.get("executed")),
            "not_executed": [row["key"] for row in rows if not row.get("executed")],
        }

        if baseline is not None:
            report["baseline"] = self._run_baseline(baseline, cases, full)

        report["detailed_runs"] = {"full": full.report}
        return report

    def _service_for(self, ablation: Ablation) -> Optional[Any]:
        if ablation.scope == "query":
            return self.service
        return self.ingestion_services.get(ablation.key)

    def _not_executed(self, ablation: Ablation) -> Dict[str, Any]:
        return {
            "key": ablation.key,
            "label": ablation.label,
            "scope": ablation.scope,
            "executed": False,
            # Names the command that would close it. A reason a reader cannot act on is
            # only marginally better than a silent zero.
            "reason": (
                "ingestion-scoped: needs a corpus re-ingested with this component disabled. "
                "Build one with scripts/build_eval_corpus.py --corpus-dir <dir> --no-layout-parser "
                "--skip-derive, then pass --ingestion-corpus %s=<dir> to scripts/run_ablations.py"
                % ablation.key
            ),
            "documented_claim_pct": ablation.documented_claim_pct,
            "primary_metric": ablation.primary_metric,
        }

    def _run_system(
        self,
        name: str,
        cases: Sequence[Dict[str, Any]],
        overrides: Optional[Dict[str, Any]],
        service: Any,
    ) -> SystemRun:
        """Execute every case against *service* and score it through the harness."""
        run = SystemRun(name=name)
        scored: List[Dict[str, Any]] = []

        for case in cases:
            query = str(case.get("query", ""))
            started = time.perf_counter()
            try:
                response = service.query(query, overrides=overrides) if overrides else service.query(query)
            except Exception as exc:
                logger.error("System %s failed on query %r: %s", name, query, exc)
                run.failures += 1
                continue
            latency_ms = (time.perf_counter() - started) * 1000.0

            scored.append(
                self.harness.evaluate_single(
                    query=query,
                    results=response.get("results", []),
                    relevant_ids=case.get("relevant_ids", []),
                    answer=response.get("answer"),
                    ground_truth_answer=case.get("ground_truth_answer"),
                    evidence=response.get("retrieved_evidence"),
                    predicted_table=case.get("predicted_table"),
                    ground_truth_table=case.get("ground_truth_table"),
                    sql_executed=(response.get("sql_execution") or {}).get("executed"),
                    latency_ms=round(latency_ms, 3),
                    stratum=case.get("stratum"),
                    latency_breakdown=response.get("latency_breakdown"),
                )
            )

        for metric in self.compared_metrics:
            run.metrics[metric] = [record.get(metric) for record in scored]

        latencies = [record.get("latency_ms", 0.0) for record in scored]
        run.report = {
            "total_queries": len(scored),
            "failures": run.failures,
            "summary_metrics": self.harness._aggregate_summary(scored, latencies),
            # The runner's own budget: it computes intervals for every system and every
            # stratum, so the default 10,000 resamples would dominate the sweep.
            "confidence_intervals": self.harness.confidence_intervals(scored, iterations=self.iterations),
            "stage_latency_ms": self.harness.stage_latency_summary(scored),
            "per_stratum": self.harness.stratified_summary(scored, iterations=self.iterations),
        }
        logger.info("Ablation system %r: %d/%d queries scored", name, len(scored), len(cases))
        return run

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _compare(
        self,
        ablation: Ablation,
        full: SystemRun,
        ablated: SystemRun,
        cross_corpus: bool = False,
    ) -> Dict[str, Any]:
        """Paired comparison of the ablated system against the full one, per metric.

        Sign convention: ``delta = ablated - full``.  Removing a component that helps
        therefore produces a negative delta on a quality metric, which is the same
        direction the documented matrix uses.

        *cross_corpus* marks a row whose ablated system answered from its own corpus.
        The answer-level metrics stay paired - the query and its ground-truth answer are
        unchanged - so they carry the measurement for that row, while the gold-region
        metrics are reported as not comparable.
        """
        comparisons: Dict[str, Any] = {}
        for metric in self.compared_metrics:
            if cross_corpus and metric in CORPUS_BOUND_METRICS:
                comparisons[metric] = {"comparable": False, "reason": NOT_COMPARABLE_REASON}
                continue
            comparison = compare_paired(
                ablated.values(metric),
                full.values(metric),
                iterations=self.iterations,
                seed=self.seed,
            )
            payload = comparison.as_dict()
            payload["mean_ablated"] = payload.pop("mean_a")
            payload["mean_full"] = payload.pop("mean_b")
            payload["relative_delta_pct"] = _relative_pct(payload["delta"], payload["mean_full"])
            comparisons[metric] = payload

        primary = comparisons.get(ablation.primary_metric)
        primary_comparable = bool(primary) and primary.get("comparable") is not False
        row: Dict[str, Any] = {
            "key": ablation.key,
            "label": ablation.label,
            "scope": ablation.scope,
            "executed": True,
            "overrides": dict(ablation.overrides),
            "rationale": ablation.rationale,
            "primary_metric": ablation.primary_metric,
            "documented_claim_pct": ablation.documented_claim_pct,
            "measured_primary_delta_pct": (primary or {}).get("relative_delta_pct"),
            "claim_supported": (
                _claim_verdict(ablation, primary) if primary_comparable else "not_comparable_across_corpora"
            ),
            "queries_scored": len(full.values(ablation.primary_metric)),
            "comparisons": comparisons,
        }
        if cross_corpus:
            row["cross_corpus"] = True
            # The published claim for this row is about a metric the row cannot measure,
            # so name the corpus-independent one that carries the result instead of
            # leaving the reader to infer which numbers survived the corpus rebuild.
            row["answer_metric"] = ANSWER_LEVEL_METRIC
            row["measured_answer_delta_pct"] = (comparisons.get(ANSWER_LEVEL_METRIC) or {}).get(
                "relative_delta_pct"
            )
            # What the two ingestions actually produced: directly comparable where the
            # gold-id metrics are not, and the evidence the row is really about.
            row["corpora"] = {
                "full": corpus_signature(self.service),
                "ablated": corpus_signature(self._service_for(ablation)),
            }
        return row

    def _run_baseline(self, baseline: Any, cases: Sequence[Dict[str, Any]], full: SystemRun) -> Dict[str, Any]:
        """Score a baseline system through the same harness and pair it against full."""
        run = self._run_system("baseline", cases, overrides=None, service=baseline)
        comparisons: Dict[str, Any] = {}
        for metric in self.compared_metrics:
            comparison = compare_paired(
                full.values(metric),
                run.values(metric),
                iterations=self.iterations,
                seed=self.seed,
            )
            payload = comparison.as_dict()
            payload["mean_full"] = payload.pop("mean_a")
            payload["mean_baseline"] = payload.pop("mean_b")
            payload["relative_delta_pct"] = _relative_pct(payload["delta"], payload["mean_baseline"])
            comparisons[metric] = payload
        return {
            "system": getattr(baseline, "__class__", type(baseline)).__name__,
            "summary_metrics": run.report.get("summary_metrics", {}),
            "full_vs_baseline": comparisons,
        }


def corpus_signature(service: Any) -> Dict[str, Any]:
    """What an ingestion produced, read from that corpus's own audit artifact.

    For an ingestion-scoped ablation this is the part of the comparison that survives
    the corpus rebuild: region and table counts are directly comparable where gold-id
    metrics are not. Returns ``{}`` when there is no audit on disk - a stub service in a
    test, or a corpus built before ``quality_audit.json`` existed - so the row reports no
    signature instead of failing the sweep.
    """
    output_dir = getattr(service, "output_dir", None)
    if not output_dir:
        return {}
    try:
        audit = json.loads((Path(output_dir) / "quality_audit.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    capabilities = audit.get("capabilities_used") or {}
    return {
        "corpus": str(output_dir),
        "total_regions": audit.get("total_regions"),
        "total_chunks": audit.get("total_chunks"),
        "structured_tables_serialized": capabilities.get("structured_tables_serialized"),
        "regions_by_layout_source": capabilities.get("regions_by_layout_source"),
        "layout_parser_enabled": capabilities.get("layout_parser_enabled"),
    }


def _relative_pct(delta: Optional[float], reference: Optional[float]) -> Optional[float]:
    """*delta* as a percentage of *reference*, or None when that is undefined.

    Returns None rather than 0.0 when the reference is zero: a relative change against
    a baseline of zero is not a number, and reporting it as 0% would read as "no
    effect" when the truth is "no denominator".
    """
    if delta is None or reference in (None, 0):
        return None
    return round((delta / reference) * 100.0, 2)


def _claim_verdict(ablation: Ablation, primary: Optional[Dict[str, Any]]) -> Optional[str]:
    """Whether the measurement supports the published delta for this row.

    Deliberately coarse - direction and significance only. Claiming agreement to one
    decimal place on a few hundred queries would repeat the mistake this runner exists
    to correct.
    """
    if primary is None or ablation.documented_claim_pct is None:
        return None
    measured = primary.get("relative_delta_pct")
    if measured is None:
        return "not_measurable"
    if not primary.get("significant"):
        return "not_significant"
    claimed_sign = 1 if ablation.documented_claim_pct > 0 else -1
    measured_sign = 1 if measured > 0 else -1
    if claimed_sign != measured_sign:
        return "contradicted"
    ratio = abs(measured) / max(1e-9, abs(ablation.documented_claim_pct))
    if 0.5 <= ratio <= 2.0:
        return "consistent_in_magnitude"
    return "same_direction_different_magnitude"


def _cross_corpus_cell(
    row: Dict[str, Any],
    answer_metric: str,
    answer_delta: Optional[float],
    answer_comparison: Dict[str, Any],
) -> str:
    """What to print for a row whose primary metric cannot cross the corpus rebuild.

    Preference order is by how much the number says. A relative answer-quality delta is
    best; an absolute one still carries a direction; failing both, the corpus census is
    the finding - "62 tables became 0" is the ablation, and it is always available.
    """
    if answer_delta is not None:
        return f"{answer_metric} {answer_delta:+.1f}%"

    absolute = answer_comparison.get("delta")
    if isinstance(absolute, (int, float)) and absolute != 0:
        return f"{answer_metric} {absolute:+.4f} abs"

    corpora = row.get("corpora") or {}
    full, ablated = corpora.get("full") or {}, corpora.get("ablated") or {}
    if full.get("total_regions") is not None and ablated.get("total_regions") is not None:
        return (
            f"tables {full.get('structured_tables_serialized')}->{ablated.get('structured_tables_serialized')}, "
            f"regions {full['total_regions']}->{ablated['total_regions']}"
        )
    return f"{answer_metric} n/a"


def markdown_matrix(report: Dict[str, Any]) -> str:
    """Render the ablation report as the matrix table, measured next to claimed."""
    lines = [
        "| Ablated component | Primary metric | Measured Δ | Claimed Δ | p | Verdict |",
        "| :--- | :--- | ---: | ---: | ---: | :--- |",
    ]
    for row in report.get("ablations", []):
        if not row.get("executed"):
            # The published claim stays in the row even when nothing measured it: dropping
            # it here would quietly retire an unverified number instead of showing it
            # standing unverified.
            claimed = row.get("documented_claim_pct")
            lines.append(
                "| {label} | {metric} | not executed | {claimed} | — | {reason} |".format(
                    label=row["label"],
                    metric=row["primary_metric"],
                    claimed="—" if claimed is None else f"{claimed:+.1f}%",
                    reason=row["reason"],
                )
            )
            continue
        primary = row["comparisons"].get(row["primary_metric"], {})
        measured = row.get("measured_primary_delta_pct")
        claimed = row.get("documented_claim_pct")
        if primary.get("comparable") is False:
            # The claim for this row is about a metric the row cannot measure. Show the
            # delta it can measure, labelled with its metric, rather than an empty cell
            # that reads as "no effect".
            answer_metric = row.get("answer_metric", ANSWER_LEVEL_METRIC)
            answer_delta = row.get("measured_answer_delta_pct")
            answer_comparison = row["comparisons"].get(answer_metric, {})
            measured_cell = _cross_corpus_cell(row, answer_metric, answer_delta, answer_comparison)
            p_cell = "n/a" if answer_comparison.get("p_value") is None else f"{answer_comparison['p_value']:.4f}"
        else:
            measured_cell = "n/a" if measured is None else f"{measured:+.1f}%"
            p_cell = "n/a" if primary.get("p_value") is None else f"{primary['p_value']:.4f}"
        lines.append(
            "| {label} | {metric} | {measured} | {claimed} | {p} | {verdict} |".format(
                label=row["label"],
                metric=row["primary_metric"],
                measured=measured_cell,
                claimed="—" if claimed is None else f"{claimed:+.1f}%",
                p=p_cell,
                verdict=row.get("claim_supported") or "n/a",
            )
        )
    return "\n".join(lines)


class AblationStudy:
    """Deprecated shim for the original delta calculator.

    Kept because ``Documentations/eval_method.md`` documents ``AblationStudy().run()``
    as the entry point. It now delegates to :class:`AblationRunner`, which needs a live
    service, so the old signature raises with instructions rather than returning a
    plausible-looking number computed from nothing.
    """

    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError(
            "AblationStudy computed a delta between two pre-computed summaries and was never called. "
            "Use AblationRunner(service, harness).run(cases), or scripts/run_ablations.py."
        )


__all__ = [
    "ABLATIONS",
    "ABLATIONS_BY_KEY",
    "ANSWER_LEVEL_METRIC",
    "COMPARED_METRICS",
    "CORPUS_BOUND_METRICS",
    "NOT_COMPARABLE_REASON",
    "Ablation",
    "AblationRunner",
    "AblationStudy",
    "SystemRun",
    "corpus_signature",
    "markdown_matrix",
]

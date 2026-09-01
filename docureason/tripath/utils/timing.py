"""timing.py — per-stage latency accounting for a single query.

The system reported one number per query: total wall time, measured *around* the
call.  Audit finding P0-5 is that this measured the harness rather than the
pipeline, and that the per-stage decomposition the evaluation spec defines
(``T_vec_search``, ``T_dense_topk``, ``T_dedup``, ``T_rerank_query`` …) existed only
as equations.  With one number, "the p95 regressed" cannot be attributed: an
embedding model reload, a cross-encoder cold start and a DuckDB failure retry are
indistinguishable.

:class:`StageTimer` accumulates elapsed milliseconds per named stage and, crucially,
reports a **residual**: total measured wall time minus the sum of the stages.  A
decomposition that does not add up is a decomposition with a hole in it, and the
residual is what makes that visible instead of letting the stages quietly describe
60% of the request.

Stage names repeat freely - three modality retrievals all contribute to
``dense_search`` - because accumulation is what makes the sum meaningful.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

#: Stages are reported in this order when present, so two runs' breakdowns line up
#: visually.  Anything not listed is appended alphabetically.
STAGE_ORDER = (
    "route",
    "dense_search",
    "sparse_search",
    "index_merge",
    "corpus_scan",
    "table_sql",
    "fuse",
    "chart_enrich",
    "parent_expand",
    "rerank",
    "generate",
    "attribute",
)


class StageTimer:
    """Accumulate per-stage elapsed time for one query.

    Not thread-safe by design for the *same* stage name: the parallel modality
    dispatch gives each worker its own timer and merges the results, which keeps the
    accounting correct without a lock on the hot path.
    """

    __slots__ = ("stages", "counts", "_started")

    def __init__(self) -> None:
        self.stages: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block and add its duration to *name*."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - start) * 1000.0)

    def add(self, name: str, elapsed_ms: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + float(elapsed_ms)
        self.counts[name] = self.counts.get(name, 0) + 1

    def merge(self, other: "StageTimer") -> None:
        """Fold another timer's stages into this one (used by the parallel dispatch)."""
        for name, elapsed in other.stages.items():
            self.stages[name] = self.stages.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + other.counts.get(name, 1)

    @property
    def elapsed_ms(self) -> float:
        """Wall time since this timer was created."""
        return (time.perf_counter() - self._started) * 1000.0

    def as_dict(self, total_ms: Optional[float] = None) -> Dict[str, Any]:
        """The breakdown, plus the total and the unaccounted-for residual.

        ``residual_ms`` can be negative when stages overlap - which is exactly what
        happens under the parallel modality dispatch, where three retrievals run
        concurrently and their summed CPU time legitimately exceeds wall time. The
        sign is the signal, so it is reported rather than clamped.
        """
        total = self.elapsed_ms if total_ms is None else float(total_ms)
        ordered = [name for name in STAGE_ORDER if name in self.stages]
        ordered += sorted(name for name in self.stages if name not in STAGE_ORDER)

        breakdown = {name: round(self.stages[name], 3) for name in ordered}
        accounted = sum(self.stages.values())
        return {
            "stages_ms": breakdown,
            "stage_calls": {name: self.counts[name] for name in ordered},
            "total_ms": round(total, 3),
            "accounted_ms": round(accounted, 3),
            "residual_ms": round(total - accounted, 3),
            "concurrent": accounted > total + 1.0,
        }


__all__ = ["StageTimer", "STAGE_ORDER"]

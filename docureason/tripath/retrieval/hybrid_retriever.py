"""hybrid_retriever.py — tri-path retrieval orchestration.

Pipeline per query:

1. Router decides which modality paths run and their fusion weights.
2. Each modality retriever queries its **FAISS + BM25S indexes** (with a substring
   scan as an explicit fallback) and returns at most ``top_k_{modality}`` candidates.
3. Reciprocal Rank Fusion merges the runs, re-applying the per-modality cap.
4. Visual candidates are enriched in place with linearized chart descriptions.
5. Parent-child expansion swaps in the parent text where one exists.
6. The cross-encoder reranks the bounded candidate set down to ``final_top_k``.

Every optional component that silently degraded now reports itself through
``last_degradation``, so a run with a stub index, a Jaccard "reranker" and a
rule-based layout parse is distinguishable from a fully functioning one.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from docureason.tripath.config import DocuReasonConfig
from docureason.tripath.utils import StageTimer, get_logger, trace_execution

from ..fusion.fuse import Fuser
from ..indexing.dense_index import DenseIndexBuilder
from ..indexing.sparse_index import BM25SIndexBuilder
from ..indexing.table_store import DuckDBTableStore
from ..ingestion.schema import Document
from ..router.configurable_router import ConfigurableRouter
from .chart_understanding import ChartUnderstandingModule
from .ranker import Ranker
from .table_retrieval import TableRetrieval
from .table_sql import TableSQLRetriever
from .text_retrieval import TextRetrieval
from .vision_retrieval import VisionRetrieval

logger = get_logger(__name__)


class HybridRetriever:
    """Index-backed tri-path retriever with fusion, chart enrichment and reranking."""

    def __init__(
        self,
        config: Optional[DocuReasonConfig] = None,
        dense: Optional[DenseIndexBuilder] = None,
        sparse: Optional[BM25SIndexBuilder] = None,
    ) -> None:
        self.config = config or DocuReasonConfig.load_preset("balanced")

        # One index-builder pair is shared by every modality retriever so the
        # embedding model and the FAISS handles are loaded once per process.
        if dense is None:
            dense = DenseIndexBuilder(output_dir=self.config.output_dir, config=self.config)
        if sparse is None:
            sparse = BM25SIndexBuilder(output_dir=self.config.output_dir, config=self.config)
        self.dense = dense
        self.sparse = sparse

        self.router = ConfigurableRouter(config=self.config)
        self.text_retrieval = TextRetrieval(dense=self.dense, sparse=self.sparse, config=self.config)
        self.table_retrieval = TableRetrieval(dense=self.dense, sparse=self.sparse, config=self.config)
        # The pipeline builds BM25S for text-bearing modalities only, so the vision
        # path has a dense run and no sparse run.
        self.vision_retrieval = VisionRetrieval(dense=self.dense, sparse=None, config=self.config)
        # The prepared table store lives beside the corpus these indexes describe.
        # When it is absent (an older run, or duckdb missing) the retriever falls back
        # to building each table in memory per query.
        self.table_store = DuckDBTableStore(self.config.output_dir)
        self.table_sql_retrieval = TableSQLRetriever(config=self.config, store=self.table_store)
        self.chart_module = ChartUnderstandingModule()
        self.fuser = Fuser(rrf_k=self.config.retrieval.rrf_k, per_run_cap=self.candidate_caps())
        self.ranker = Ranker(config=self.config)

        #: Degradation labels and retrieval statistics from the most recent query.
        self.last_degradation: List[str] = []
        self.last_stats: Dict[str, Any] = {}
        #: SQL candidates from the most recent query, so the serving layer can build
        #: its SQL payload without running the DuckDB path a second time.
        self.last_sql_results: List[Dict[str, Any]] = []
        #: Per-stage timings from the most recent query, for the response's
        #: ``latency_breakdown``.  Owned here rather than in the serving layer because
        #: this is where the stages are.
        self.last_timer: StageTimer = StageTimer()

    def candidate_caps(self) -> Dict[str, int]:
        """Per-modality candidate caps read from ``RetrievalConfig``."""
        retrieval = self.config.retrieval
        return {
            "text": int(retrieval.top_k_text),
            "table": int(retrieval.top_k_table),
            "vision": int(retrieval.top_k_vision),
            "default": int(retrieval.top_k_text),
        }

    @trace_execution(logger=logger)
    def retrieve(
        self,
        query: str,
        documents: List[Document],
        timer: Optional[StageTimer] = None,
    ) -> List[Dict[str, Any]]:
        timer = timer or StageTimer()
        self.last_timer = timer

        with timer.stage("route"):
            route_flags = self.router.route(query)
            router_weights = self.router.get_route_weights(query)
        logger.info("HybridRetriever router flags for query '%s': %s (weights: %s)", query, route_flags, router_weights)

        modality_modes: Dict[str, Any] = {}
        self.last_sql_results = []

        # 1. Modality retrieval, each capped at top_k_{modality}.  The paths are
        #    independent, so they run concurrently when enabled; results are collected
        #    in the fixed order the tasks were declared, which keeps fusion input -
        #    and therefore ranking - identical either way.
        tasks = self._build_tasks(route_flags, query, documents)
        outcomes = self._dispatch(tasks)

        batches: List[List[Dict[str, Any]]] = []
        for name, batch, stats, elapsed_ms in outcomes:
            batches.append(batch)
            if name == "table_sql":
                self.last_sql_results = batch
                # The SQL path reports no sub-steps of its own, so its wall time is
                # the stage. The modality paths are accounted by the dense/sparse/merge
                # timings they report; whatever they spend outside those shows up in
                # the timer's residual rather than being invented here.
                timer.add("table_sql", elapsed_ms)
            else:
                modality_modes[name] = stats
                self._fold_path_timings(timer, stats)

        pre_fusion = sum(len(batch) for batch in batches)

        # 2. Reciprocal Rank Fusion, re-applying the caps.
        with timer.stage("fuse"):
            fused = self.fuser.fuse(batches, router_weights=router_weights)

        # 3. Chart enrichment, in place — this used to append a second copy of every
        #    visual candidate, which is why the duplicate-rank ratio was so high.
        with timer.stage("chart_enrich"):
            candidates = self.chart_module.understand(query, fused)

        # 4. Parent-child expansion (config-gated; the flag was never checked before).
        if self.config.reranker.enable_parent_expansion:
            with timer.stage("parent_expand"):
                candidates = self._expand_parents(candidates)

        # 5. Cross-encoder reranking down to final_top_k.
        with timer.stage("rerank"):
            reranked = self.ranker.rank(query, candidates)
        top_k = self.config.reranker.final_top_k
        final = reranked[:top_k] if top_k > 0 else reranked

        self.last_stats = {
            "modality_modes": modality_modes,
            "pre_fusion_candidates": pre_fusion,
            "fused_candidates": len(fused),
            "reranked_candidates": len(reranked),
            "returned": len(final),
            "charts_enriched": self.chart_module.charts_enriched,
            "sql_candidates": len(self.last_sql_results),
            "parallel_paths": self._parallel_enabled() and len(tasks) > 1,
            "paths_run": [name for name, *_ in tasks],
        }
        self.last_degradation = self._collect_degradation(modality_modes, final)
        if self.last_degradation:
            logger.warning("HybridRetriever degradation: %s", ", ".join(self.last_degradation))
        return final

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parallel_enabled(self) -> bool:
        serving = getattr(self.config, "serving", None)
        return bool(getattr(serving, "enable_parallel_paths", True))

    def _build_tasks(
        self,
        route_flags: Dict[str, bool],
        query: str,
        documents: List[Document],
    ) -> List[Tuple[str, Callable[[], List[Dict[str, Any]]], Any]]:
        """The retrieval calls this query needs, in a fixed, deterministic order.

        Each entry is ``(name, thunk, stats_source)``.  ``stats_source`` is the
        retriever whose ``last_stats`` describes the call, read only after the thunk
        has returned.
        """
        tasks: List[Tuple[str, Callable[[], List[Dict[str, Any]]], Any]] = []

        if route_flags.get("text", True):
            tasks.append(("text", lambda: self.text_retrieval.retrieve(query, documents), self.text_retrieval))

        if route_flags.get("table", True):
            tasks.append(("table", lambda: self.table_retrieval.retrieve(query, documents), self.table_retrieval))
            # Ablation A2 ("remove the DuckDB text-to-SQL retriever") switches the
            # whole SQL path off rather than letting it run and discarding the rows,
            # so the measured delta includes its latency as well as its accuracy.
            if self.config.retrieval.enable_table_sql:
                tasks.append(("table_sql", lambda: self.table_sql_retrieval.retrieve(query, documents), None))

        if route_flags.get("vision", True):
            tasks.append(("vision", lambda: self.vision_retrieval.retrieve(query, documents), self.vision_retrieval))

        return tasks

    def _dispatch(
        self,
        tasks: List[Tuple[str, Callable[[], List[Dict[str, Any]]], Any]],
    ) -> List[Tuple[str, List[Dict[str, Any]], Any, float]]:
        """Run *tasks*, concurrently when enabled, returning results in task order.

        A failing path must not take the query down with it: the remaining paths still
        answer, and the failure is logged and surfaces as a missing modality in
        ``modality_modes`` rather than a 500.  Sequential mode raises nothing extra
        either, so the two dispatch modes fail the same way.
        """
        if not tasks:
            return []

        def run(index: int) -> Tuple[str, List[Dict[str, Any]], Any, float]:
            name, thunk, source = tasks[index]
            started = time.perf_counter()
            try:
                batch = thunk() or []
            except Exception as exc:  # one dead path, not a dead query
                elapsed = (time.perf_counter() - started) * 1000.0
                logger.error("Retrieval path %r failed: %s", name, exc)
                return name, [], {"mode": "failed", "error": str(exc)}, elapsed
            elapsed = (time.perf_counter() - started) * 1000.0
            stats = getattr(source, "last_stats", None) if source is not None else None
            return name, batch, stats, elapsed

        if not self._parallel_enabled() or len(tasks) == 1:
            return [run(i) for i in range(len(tasks))]

        # max_workers is bounded by the task count: the paths are I/O- and
        # torch-bound (both release the GIL), so more threads than paths buys nothing.
        with ThreadPoolExecutor(max_workers=len(tasks), thread_name_prefix="docureason-path") as pool:
            return list(pool.map(run, range(len(tasks))))

    @staticmethod
    def _fold_path_timings(timer: StageTimer, stats: Any) -> None:
        """Add a retriever's self-reported sub-step timings to the query timer.

        Under parallel dispatch these sum to more than the wall time they occupied,
        which is why :meth:`StageTimer.as_dict` reports a signed residual instead of
        pretending the stages partition the request.
        """
        if not isinstance(stats, dict):
            return
        for stat_key, stage in (
            ("dense_ms", "dense_search"),
            ("sparse_ms", "sparse_search"),
            ("index_merge_ms", "index_merge"),
            ("corpus_scan_ms", "corpus_scan"),
        ):
            value = stats.get(stat_key)
            if value:
                timer.add(stage, float(value))

    @staticmethod
    def _expand_parents(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Swap in the parent text when it is strictly longer than the child text."""
        expanded: List[Dict[str, Any]] = []
        for candidate in candidates:
            item = dict(candidate)
            parent_text = candidate.get("parent_text")
            if parent_text and len(parent_text) > len(candidate.get("text", "")):
                item["child_text"] = candidate.get("text", "")
                item["text"] = parent_text
                item["parent_expanded"] = True
            expanded.append(item)
        return expanded

    def _collect_degradation(
        self,
        modality_modes: Dict[str, Any],
        final: List[Dict[str, Any]],
    ) -> List[str]:
        """Enumerate every fallback that fired, as short ``subsystem:reason`` labels."""
        labels: List[str] = []

        for modality, stats in modality_modes.items():
            mode = (stats or {}).get("mode")
            if mode == "substring_scan":
                labels.append(f"retrieval:{modality}:substring_scan")
            if (stats or {}).get("stale_hits"):
                labels.append(f"index:{modality}:stale_hits={stats['stale_hits']}")

        for modality in ("text", "table", "vision"):
            if modality in modality_modes and self.dense.ntotal(modality) == 0:
                labels.append(f"dense_index:{modality}:empty")

        modes = {item.get("reranker_mode") for item in final if item.get("reranker_mode")}
        for mode in sorted(m for m in modes if m and m != "cross_encoder"):
            labels.append(f"reranker:{mode}")

        # Ablations are deliberate, but they still change what the response means, so
        # they are reported through the same channel as an accidental degradation.
        if not self.config.retrieval.enable_sparse_retrieval:
            labels.append("ablation:sparse_retrieval_disabled")
        if not self.config.retrieval.enable_table_sql:
            labels.append("ablation:table_sql_disabled")
        if not self.config.reranker.enable_parent_expansion:
            labels.append("ablation:parent_expansion_disabled")
        if not self.config.reranker.enable_heading_penalty:
            labels.append("ablation:heading_penalty_disabled")
        if not self.config.retrieval.enable_index_retrieval:
            labels.append("ablation:index_retrieval_disabled")

        sql_modes = {item.get("mode") for item in self.last_sql_results}
        if "sql_simulation" in sql_modes:
            labels.append("table_sql:python_runner")

        return labels

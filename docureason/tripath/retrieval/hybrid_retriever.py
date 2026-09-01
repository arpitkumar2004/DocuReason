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

from typing import Any, Dict, List, Optional

from docureason.tripath.config import DocuReasonConfig
from docureason.tripath.utils import get_logger, trace_execution

from ..fusion.fuse import Fuser
from ..indexing.dense_index import DenseIndexBuilder
from ..indexing.sparse_index import BM25SIndexBuilder
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
        self.table_sql_retrieval = TableSQLRetriever()
        self.chart_module = ChartUnderstandingModule()
        self.fuser = Fuser(rrf_k=self.config.retrieval.rrf_k, per_run_cap=self.candidate_caps())
        self.ranker = Ranker(config=self.config)

        #: Degradation labels and retrieval statistics from the most recent query.
        self.last_degradation: List[str] = []
        self.last_stats: Dict[str, Any] = {}
        #: SQL candidates from the most recent query, so the serving layer can build
        #: its SQL payload without running the DuckDB path a second time.
        self.last_sql_results: List[Dict[str, Any]] = []

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
    def retrieve(self, query: str, documents: List[Document]) -> List[Dict[str, Any]]:
        route_flags = self.router.route(query)
        router_weights = self.router.get_route_weights(query)
        logger.info("HybridRetriever router flags for query '%s': %s (weights: %s)", query, route_flags, router_weights)

        batches: List[List[Dict[str, Any]]] = []
        modality_modes: Dict[str, Any] = {}
        self.last_sql_results = []

        # 1. Modality retrieval, each capped at top_k_{modality}.
        if route_flags.get("text", True):
            batches.append(self.text_retrieval.retrieve(query, documents))
            modality_modes["text"] = self.text_retrieval.last_stats

        if route_flags.get("table", True):
            batches.append(self.table_retrieval.retrieve(query, documents))
            modality_modes["table"] = self.table_retrieval.last_stats
            self.last_sql_results = self.table_sql_retrieval.retrieve(query, documents)
            batches.append(self.last_sql_results)

        if route_flags.get("vision", True):
            batches.append(self.vision_retrieval.retrieve(query, documents))
            modality_modes["vision"] = self.vision_retrieval.last_stats

        pre_fusion = sum(len(batch) for batch in batches)

        # 2. Reciprocal Rank Fusion, re-applying the caps.
        fused = self.fuser.fuse(batches, router_weights=router_weights)

        # 3. Chart enrichment, in place — this used to append a second copy of every
        #    visual candidate, which is why the duplicate-rank ratio was so high.
        candidates = self.chart_module.understand(query, fused)

        # 4. Parent-child expansion (config-gated; the flag was never checked before).
        if self.config.reranker.enable_parent_expansion:
            candidates = self._expand_parents(candidates)

        # 5. Cross-encoder reranking down to final_top_k.
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
        }
        self.last_degradation = self._collect_degradation(modality_modes, final)
        if self.last_degradation:
            logger.warning("HybridRetriever degradation: %s", ", ".join(self.last_degradation))
        return final

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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

        sql_modes = {item.get("mode") for item in self.last_sql_results}
        if "sql_simulation" in sql_modes:
            labels.append("table_sql:python_runner")

        return labels

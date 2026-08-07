from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.tripath.config import DocuReasonConfig
from src.tripath.utils import get_logger, trace_execution

from ..fusion.fuse import Fuser
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
    """Hybrid Retriever with Step 2 Parent-Child Expansion and Step 4 Agent Reranking."""

    def __init__(self, config: Optional[DocuReasonConfig] = None) -> None:
        self.config = config or DocuReasonConfig.load_preset("balanced")
        self.router = ConfigurableRouter(config=self.config)
        self.text_retrieval = TextRetrieval()
        self.table_retrieval = TableRetrieval()
        self.vision_retrieval = VisionRetrieval()
        self.table_sql_retrieval = TableSQLRetriever()
        self.chart_module = ChartUnderstandingModule()
        self.fuser = Fuser(rrf_k=self.config.retrieval.rrf_k)
        self.ranker = Ranker(config=self.config)

    @trace_execution(logger=logger)
    def retrieve(self, query: str, documents: List[Document]) -> List[Dict[str, Any]]:
        # 1. Multi-Modal Router Decision & Weights
        route_flags = self.router.route(query)
        router_weights = self.router.get_route_weights(query)
        logger.info("HybridRetriever router flags for query '%s': %s (weights: %s)", query, route_flags, router_weights)

        batches: List[List[Dict[str, Any]]] = []

        # 2. Modality Retrieval Execution
        if route_flags.get("text", True):
            batches.append(self.text_retrieval.retrieve(query, documents))

        if route_flags.get("table", True):
            batches.append(self.table_retrieval.retrieve(query, documents))
            batches.append(self.table_sql_retrieval.retrieve(query, documents))

        if route_flags.get("vision", True):
            batches.append(self.vision_retrieval.retrieve(query, documents))

        # 3. Reciprocal Rank Fusion (RRF)
        fused = self.fuser.fuse(batches, router_weights=router_weights)

        # 4. Chart Understanding Evidence Augmentation
        chart_evidence = self.chart_module.understand(query, fused)
        all_candidates = fused + chart_evidence

        # Step 2: Parent-Child Recursive Expansion (Expand hits to full Parent Chunk text)
        expanded_candidates: List[Dict[str, Any]] = []
        for cand in all_candidates:
            expanded_cand = dict(cand)
            parent_text = cand.get("parent_text")
            if parent_text and len(parent_text) > len(cand.get("text", "")):
                # Keep original child text in metadata, use parent_text for context & generation
                expanded_cand["child_text"] = cand.get("text", "")
                expanded_cand["text"] = parent_text
            expanded_candidates.append(expanded_cand)

        # Step 4: Cross-Encoder Agent Reranking
        reranked = self.ranker.rank(query, expanded_candidates)
        top_k = self.config.reranker.final_top_k
        return reranked[:top_k] if top_k > 0 else reranked

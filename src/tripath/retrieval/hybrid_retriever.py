from __future__ import annotations

from typing import List

from ..ingestion.schema import Document
from .text_retrieval import TextRetrieval
from .table_retrieval import TableRetrieval
from .vision_retrieval import VisionRetrieval
from .table_sql import TableSQLRetriever
from .chart_understanding import ChartUnderstandingModule
from ..fusion.normalize import Normalizer
from ..fusion.fuse import Fuser


class HybridRetriever:
    """A lightweight hybrid retriever that combines text, table, and vision evidence."""

    def __init__(self) -> None:
        self.text_retrieval = TextRetrieval()
        self.table_retrieval = TableRetrieval()
        self.vision_retrieval = VisionRetrieval()
        self.table_sql_retrieval = TableSQLRetriever()
        self.chart_module = ChartUnderstandingModule()

    def retrieve(self, query: str, documents: List[Document]) -> List[dict]:
        batches = []
        batches.append(self.text_retrieval.retrieve(query, documents))
        batches.append(self.table_retrieval.retrieve(query, documents))
        batches.append(self.vision_retrieval.retrieve(query, documents))
        batches.append(self.table_sql_retrieval.retrieve(query, documents))

        fused = Fuser().fuse(batches)
        normalized = Normalizer().normalize(fused)
        chart_evidence = self.chart_module.understand(query, normalized)
        return normalized + chart_evidence

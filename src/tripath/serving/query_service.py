from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..ingestion.docling_wrapper import DoclingWrapper
from ..retrieval.embedder import ChunkEmbedder
from ..retrieval.ranker import Ranker
from ..retrieval.text_retrieval import TextRetrieval
from ..retrieval.table_retrieval import TableRetrieval
from ..retrieval.vision_retrieval import VisionRetrieval
from ..router.configurable_router import ConfigurableRouter
from ..fusion.fuse import Fuser
from ..generation.generate import GenerationModule


class QueryService:
    """A small end-to-end Phase 2 query service for sample documents."""

    def __init__(self, input_dir: str | Path, output_dir: str | Path) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.documents = DoclingWrapper(input_dir=self.input_dir, output_dir=self.output_dir).ingest()
        self.embedder = ChunkEmbedder()
        self.router = ConfigurableRouter()
        self.ranker = Ranker()
        self.generator = GenerationModule()

    def query(self, text: str) -> Dict[str, object]:
        route = self.router.route(text)
        results = []
        if route.get("text"):
            results.append(TextRetrieval().retrieve(text, self.documents))
        if route.get("table"):
            results.append(TableRetrieval().retrieve(text, self.documents))
        if route.get("vision"):
            results.append(VisionRetrieval().retrieve(text, self.documents))

        fused = Fuser().fuse(results) if results else []
        ranked = self.ranker.rank(text, fused)
        embeddings = self.embedder.embed([chunk for document in self.documents for chunk in document.chunks])
        answer = self.generator.generate(text, ranked)
        return {
            "query": text,
            "route": route,
            "results": ranked,
            "embeddings": embeddings,
            "answer": answer,
        }

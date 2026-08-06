from pathlib import Path
from typing import Any, Dict, List, Optional
from src.tripath.config import DocuReasonConfig
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
    """An end-to-end multimodal query service for enterprise documents."""

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        config: Optional[DocuReasonConfig] = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or DocuReasonConfig.load_from_yaml("configs/config.yaml")
        self.config.output_dir = str(self.output_dir)

        self.documents = DoclingWrapper(input_dir=self.input_dir, output_dir=self.output_dir).ingest()
        self.embedder = ChunkEmbedder()
        self.router = ConfigurableRouter(config=self.config)
        self.ranker = Ranker(config=self.config)
        self.generator = GenerationModule(config=self.config)

    def query(self, text: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, object]:
        effective_config = self.config.apply_query_overrides(overrides or {})
        router = ConfigurableRouter(config=effective_config) if overrides else self.router
        ranker = Ranker(config=effective_config) if overrides else self.ranker
        generator = GenerationModule(config=effective_config) if overrides else self.generator

        route = router.route(text)
        results = []
        if route.get("text"):
            results.append(TextRetrieval().retrieve(text, self.documents))
        if route.get("table"):
            results.append(TableRetrieval().retrieve(text, self.documents))
        if route.get("vision"):
            results.append(VisionRetrieval().retrieve(text, self.documents))

        fused = Fuser(rrf_k=effective_config.retrieval.rrf_k).fuse(results) if results else []
        ranked = ranker.rank(text, fused)
        top_k = effective_config.reranker.final_top_k
        final_ranked = ranked[:top_k] if top_k > 0 else ranked

        embeddings = self.embedder.embed([chunk for document in self.documents for chunk in document.chunks])
        answer = generator.generate(text, final_ranked)
        return {
            "query": text,
            "route": route,
            "results": final_ranked,
            "embeddings": embeddings,
            "answer": answer,
            "preset": effective_config.preset,
        }

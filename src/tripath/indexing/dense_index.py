"""dense_index.py — FAISS-based dense vector index for document retrieval.

Builds one FAISS index per modality (text, table, vision) with support for
domain-specific embedding presets and HNSW graph parameter tuning.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.tripath.config import DocuReasonConfig, IndexingConfig
from src.tripath.utils import get_logger, log_pipeline_flag, trace_execution

logger = get_logger(__name__)

_DOMAIN_PRESETS: Dict[str, str] = {
    "general": "sentence-transformers/all-MiniLM-L6-v2",
    "biomedical": "pritamdeka/S-PubMedBert-MS-MARCO",
    "medical": "pritamdeka/S-PubMedBert-MS-MARCO",
    "legal": "law-ai/InLegalBERT",
    "financial": "ProsusAI/finbert",
    "finance": "ProsusAI/finbert",
    "code": "flax-sentence-embeddings/st-codesearch-distilroberta-base",
    "technical": "flax-sentence-embeddings/st-codesearch-distilroberta-base",
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_DIM = 384


class DenseIndexBuilder:
    """Build and query per-modality FAISS dense indices with domain presets and HNSW parameters."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "artifacts",
        model_name: Optional[str] = None,
        domain: str = "general",
        index_type: str = "hnsw",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        config: Optional[Union[IndexingConfig, DocuReasonConfig]] = None,
    ) -> None:
        if isinstance(config, DocuReasonConfig):
            idx_cfg = config.indexing
            self.output_dir = Path(config.output_dir)
        elif isinstance(config, IndexingConfig):
            idx_cfg = config
            self.output_dir = Path(output_dir)
        else:
            idx_cfg = None
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if idx_cfg:
            self.domain = idx_cfg.domain.lower()
            self.model_name = idx_cfg.model_name or _DOMAIN_PRESETS.get(self.domain, _DOMAIN_PRESETS["general"])
            self.index_type = idx_cfg.index_type.lower()
            self.hnsw_m = idx_cfg.hnsw_m
            self.hnsw_ef_construction = idx_cfg.hnsw_ef_construction
            self.hnsw_ef_search = idx_cfg.hnsw_ef_search
        else:
            self.domain = domain.lower()
            self.model_name = model_name or _DOMAIN_PRESETS.get(self.domain, _DOMAIN_PRESETS["general"])
            self.index_type = index_type.lower()
            self.hnsw_m = hnsw_m
            self.hnsw_ef_construction = hnsw_ef_construction
            self.hnsw_ef_search = hnsw_ef_search

        self._model: Any = None  # lazy-loaded SentenceTransformer
        log_pipeline_flag("dense_index_model", self.model_name, f"FAISS model (domain={self.domain})", logger)
        log_pipeline_flag("dense_index_type", self.index_type, f"FAISS index type (hnsw_m={self.hnsw_m})", logger)

    @trace_execution(logger=logger)
    def build(self, records: List[Dict[str, Any]], modality: str) -> Path:
        """Encode *records* and write a FAISS index to disk."""
        index_path = self.output_dir / f"faiss_{modality}.index"
        meta_path = self.output_dir / f"faiss_{modality}_meta.json"

        if not records:
            logger.warning("DenseIndexBuilder: no records for modality=%s", modality)
            meta_path.write_text("[]", encoding="utf-8")
            return index_path

        texts = [r.get("text", "") or "" for r in records]
        meta = [
            {
                "id": r.get("id", str(i)),
                "document_id": r.get("document_id", ""),
                "modality": r.get("modality", modality),
                "metadata": r.get("metadata", {}),
                "text": texts[i][:500],
            }
            for i, r in enumerate(records)
        ]

        try:
            import faiss

            model = self._get_model()
            if model is None:
                raise ImportError("sentence_transformers unavailable")

            logger.info("DenseIndexBuilder: encoding %d records (modality=%s)", len(texts), modality)
            vectors = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,  # cosine via inner product
            ).astype("float32")

            dim = vectors.shape[1] if hasattr(vectors, "shape") and len(vectors.shape) > 1 else _DIM

            if self.index_type == "hnsw" and hasattr(faiss, "IndexHNSWFlat"):
                logger.info(
                    "DenseIndexBuilder: building IndexHNSWFlat (dim=%d, M=%d, efConstruction=%d)",
                    dim, self.hnsw_m, self.hnsw_ef_construction,
                )
                index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = self.hnsw_ef_construction
            else:
                index = faiss.IndexFlatIP(dim)

            index.add(vectors)
            faiss.write_index(index, str(index_path))
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("DenseIndexBuilder: wrote %s (%d vectors, type=%s)", index_path.name, len(vectors), self.index_type)

        except Exception as exc:
            logger.warning("DenseIndexBuilder: FAISS build failed (%s) — writing stub", exc)
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            index_path.write_bytes(b"")

        return index_path

    def search(self, query: str, modality: str, k: int = 5) -> List[Dict[str, Any]]:
        """Query the FAISS index for *modality* and return top-*k* results."""
        index_path = self.output_dir / f"faiss_{modality}.index"
        meta_path = self.output_dir / f"faiss_{modality}_meta.json"

        if not index_path.exists() or index_path.stat().st_size == 0:
            logger.debug("DenseIndexBuilder.search: index not found for %s", modality)
            return []

        try:
            import faiss

            model = self._get_model()
            if model is None:
                return []

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            index = faiss.read_index(str(index_path))

            if hasattr(index, "hnsw"):
                index.hnsw.efSearch = self.hnsw_ef_search

            q_vec = model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")

            distances, indices = index.search(q_vec, min(k, index.ntotal))
            results = []
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(meta):
                    continue
                entry = dict(meta[idx])
                entry["score"] = round(float(score), 4)
                results.append(entry)
            return results

        except Exception as exc:
            logger.warning("DenseIndexBuilder.search failed: %s", exc)
            return []

    def _get_model(self) -> Optional[Any]:
        """Lazy-load and cache the SentenceTransformer model."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer loaded: %s", self.model_name)
        except Exception as exc:
            logger.warning("Failed to load SentenceTransformer (%s) — using fallback", exc)
            self._model = None
        return self._model

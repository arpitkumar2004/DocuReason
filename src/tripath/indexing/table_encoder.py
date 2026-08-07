from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.tripath.utils import get_logger, trace_execution

from ..ingestion.schema import Document

logger = get_logger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TableEncoder:
    """Encode table regions with sentence-transformers over GFM Markdown text.

    Payload includes ``table_json`` (columns + rows schema) for the SQL
    retrieval path and ``table_markdown`` for semantic similarity.
    Falls back to hash-vectors when ``sentence_transformers`` is absent.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any = None

    @trace_execution(logger=logger)
    def encode(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Return table records ready for DenseIndexBuilder and BM25SIndexBuilder."""
        records: List[Dict[str, Any]] = []
        texts_pending: List[str] = []
        indices_pending: List[int] = []

        for document in documents:
            for region in document.regions:
                if region.type == "table":
                    # Prefer the clean Markdown over raw text for embedding.
                    embed_text = region.table_markdown or region.text or ""
                    records.append({
                        "id": f"{document.id}-table-{len(records)}",
                        "document_id": document.id,
                        "region_type": "table",
                        "modality": "table",
                        "text": embed_text,
                        "table_markdown": region.table_markdown or "",
                        "table_json": json.dumps(region.table_json, ensure_ascii=False)
                            if region.table_json else "",
                        "page_no": region.page_no,
                        "bbox": list(region.bbox) if region.bbox else None,
                        "layout_source": region.layout_source,
                        "vector": [],
                    })
                    texts_pending.append(embed_text)
                    indices_pending.append(len(records) - 1)

        if not texts_pending:
            return records

        vectors = self._embed(texts_pending)
        for list_idx, record_idx in enumerate(indices_pending):
            records[record_idx]["vector"] = vectors[list_idx]

        return records

    def _embed(self, texts: List[str]) -> List[List[float]]:
        try:
            model = self._get_model()
            if model is not None:
                vecs = model.encode(
                    texts,
                    batch_size=32,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return [v.tolist() for v in vecs]
        except Exception as exc:
            logger.debug("TableEncoder embedding failed: %s", exc)
        return [self._hash_vector(t) for t in texts]

    def _get_model(self) -> Optional[Any]:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = None
        return self._model

    @staticmethod
    def _hash_vector(text: str) -> List[float]:
        import hashlib
        tokens = sorted(set(text.lower().split()))
        return [
            round(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) / 0xFFFFFF, 4)
            for t in tokens
        ]

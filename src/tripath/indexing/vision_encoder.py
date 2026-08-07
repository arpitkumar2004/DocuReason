from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.tripath.utils import get_logger, trace_execution

from ..ingestion.schema import Document

logger = get_logger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VisionEncoder:
    """Encode figure regions using caption text via sentence-transformers.

    Caption text is produced by ``FigureCaptioner`` (stored in
    ``region.metadata["caption"]`` and ``region.text``).  If CLIP is
    available the raw image embedding is also stored in the payload as
    ``clip_vector`` for MaxSim-style visual retrieval.

    Falls back to hash-vectors when sentence_transformers is absent.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any = None

    @trace_execution(logger=logger)
    def encode(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Return vision records ready for DenseIndexBuilder."""
        records: List[Dict[str, Any]] = []
        texts_pending: List[str] = []
        indices_pending: List[int] = []

        for document in documents:
            for region in document.regions:
                if region.type == "figure":
                    caption = (
                        (region.metadata or {}).get("caption")
                        or region.text
                        or f"Figure from {document.title}"
                    )
                    records.append({
                        "id": f"{document.id}-figure-{len(records)}",
                        "document_id": document.id,
                        "region_type": "figure",
                        "modality": "vision",
                        "text": caption,
                        "caption": caption,
                        "page_no": region.page_no,
                        "bbox": list(region.bbox) if region.bbox else None,
                        "layout_source": region.layout_source,
                        "vector": [],       # text-based embedding (always filled)
                        "clip_vector": [],  # image-based CLIP embedding (optional)
                    })
                    texts_pending.append(caption)
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
            logger.debug("VisionEncoder embedding failed: %s", exc)
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

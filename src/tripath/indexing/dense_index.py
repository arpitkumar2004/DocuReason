"""dense_index.py — FAISS-based dense vector index for Phase 1 retrieval.

Builds one ``faiss.IndexFlatIP`` index per modality (text, table, vision)
from ``sentence-transformers/all-MiniLM-L6-v2`` embeddings (dim=384).

Files produced per modality
----------------------------
* ``faiss_{modality}.index`` — FAISS index binary
* ``faiss_{modality}_meta.json`` — id→metadata mapping for result hydration

Fallback behaviour
------------------
If ``faiss`` or ``sentence_transformers`` are not installed the builder
writes a lightweight JSON stub so callers always get a consistent
``search()`` interface (returns empty results instead of raising).

Usage
-----
::

    from src.tripath.indexing.dense_index import DenseIndexBuilder

    builder = DenseIndexBuilder(output_dir=Path("artifacts/output"))
    builder.build(records, modality="text")
    results = builder.search("quarterly revenue", modality="text", k=5)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tripath.utils import get_logger, log_pipeline_flag, trace_execution

logger = get_logger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_DIM = 384


class DenseIndexBuilder:
    """Build and query per-modality FAISS dense indices.

    Parameters
    ----------
    output_dir:
        Directory where ``faiss_*.index`` and ``faiss_*_meta.json`` are saved.
    model_name:
        Sentence-transformers model name / HF path.
    """

    def __init__(
        self,
        output_dir: str | Path = "artifacts",
        model_name: str = _MODEL_NAME,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model: Any = None  # lazy-loaded SentenceTransformer
        log_pipeline_flag("dense_index_model", model_name, "FAISS embedding model name", logger)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @trace_execution(logger=logger)
    def build(self, records: List[Dict[str, Any]], modality: str) -> Path:
        """Encode *records* and write a FAISS index to disk.

        Parameters
        ----------
        records:
            List of dicts with at minimum ``{"id": str, "text": str}``.
        modality:
            One of ``"text"``, ``"table"``, ``"vision"``.  Used as the
            filename stem.

        Returns
        -------
        Path to the written ``.index`` file.
        """
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
                "text": texts[i][:500],  # keep a snippet for hydration
            }
            for i, r in enumerate(records)
        ]

        try:
            import faiss
            import numpy as np

            model = self._get_model()
            if model is None:
                raise ImportError("sentence_transformers unavailable")

            logger.info(
                "DenseIndexBuilder: encoding %d records (modality=%s)",
                len(texts), modality,
            )
            vectors = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,  # cosine via inner product
            ).astype("float32")

            index = faiss.IndexFlatIP(_DIM)
            index.add(vectors)  # type: ignore[arg-type]
            faiss.write_index(index, str(index_path))
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(
                "DenseIndexBuilder: wrote %s (%d vectors)", index_path.name, len(vectors)
            )

        except Exception as exc:
            logger.warning(
                "DenseIndexBuilder: FAISS build failed (%s) — writing stub", exc
            )
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            # Write a stub so the path always exists.
            index_path.write_bytes(b"")

        return index_path

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        modality: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query the FAISS index for *modality* and return top-*k* results.

        Returns an empty list if the index doesn't exist or FAISS is not
        installed.
        """
        index_path = self.output_dir / f"faiss_{modality}.index"
        meta_path = self.output_dir / f"faiss_{modality}_meta.json"

        if not index_path.exists() or index_path.stat().st_size == 0:
            logger.debug("DenseIndexBuilder.search: index not found for %s", modality)
            return []

        try:
            import faiss
            import numpy as np

            model = self._get_model()
            if model is None:
                return []

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            index = faiss.read_index(str(index_path))

            q_vec = model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")

            distances, indices = index.search(q_vec, min(k, index.ntotal))  # type: ignore
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

    # ------------------------------------------------------------------
    # Model loader
    # ------------------------------------------------------------------

    def _get_model(self) -> Optional[Any]:
        """Lazy-load and cache the SentenceTransformer model."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer loaded: %s", self.model_name)
        except ImportError:
            logger.warning(
                "sentence_transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self._model = None
        except Exception as exc:
            logger.warning("Failed to load SentenceTransformer: %s", exc)
            self._model = None
        return self._model

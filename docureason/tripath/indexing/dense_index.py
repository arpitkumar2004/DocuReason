"""dense_index.py — FAISS-based dense vector index for document retrieval.

Builds one FAISS index per modality (text, table, vision) with support for
domain-specific embedding presets and HNSW graph parameter tuning.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from docureason.tripath.config import DocuReasonConfig, IndexingConfig
from docureason.tripath.model_pins import resolve_revision
from docureason.tripath.utils import get_logger, log_pipeline_flag, trace_execution

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


class IndexBuildError(RuntimeError):
    """Raised when a FAISS index could not be built despite its dependencies being present.

    A silently-written empty index is the worst possible failure shape: at query
    time it returns ``[]``, which is indistinguishable from "no relevant
    documents".  Genuine build failures therefore raise; a *missing optional
    dependency* is reported as a degradation instead (see ``build_stats``).
    """


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
            self.model_revision = resolve_revision(self.model_name, getattr(idx_cfg, "model_revision", None))
            self.hnsw_min_vectors = getattr(idx_cfg, "hnsw_min_vectors", 10000)
        else:
            self.domain = domain.lower()
            self.model_name = model_name or _DOMAIN_PRESETS.get(self.domain, _DOMAIN_PRESETS["general"])
            self.index_type = index_type.lower()
            self.hnsw_m = hnsw_m
            self.hnsw_ef_construction = hnsw_ef_construction
            self.hnsw_ef_search = hnsw_ef_search
            self.model_revision = resolve_revision(self.model_name)
            self.hnsw_min_vectors = 10000

        #: Per-modality build observations: {"records", "ntotal", "degraded"}.
        #: ``degraded`` is None on success and a short reason string otherwise.
        self.build_stats: Dict[str, Dict[str, Any]] = {}
        self._model: Any = None  # lazy-loaded SentenceTransformer
        #: Per-modality FAISS index cache so search() does not re-read from disk per query.
        self._index_cache: Dict[str, Any] = {}
        #: Per-modality metadata cache. search() used to json.loads the whole sidecar on
        #: every query, which on a large corpus cost more than the vector search itself.
        self._meta_cache: Dict[str, List[Dict[str, Any]]] = {}
        #: Guards the lazy model load and the two caches. The parallel modality
        #: dispatch calls search() from three threads against this one builder, and
        #: without the lock all three could load the embedding model concurrently.
        self._load_lock = threading.Lock()
        log_pipeline_flag("dense_index_model", self.model_name, f"FAISS model (domain={self.domain})", logger)
        log_pipeline_flag("dense_index_type", self.index_type, f"FAISS index type (hnsw_m={self.hnsw_m})", logger)
        log_pipeline_flag(
            "dense_index_revision",
            self.model_revision or "hub-default",
            "Pinned embedding weights" if self.model_revision else "UNPINNED - hub may change these weights",
            logger,
        )

    @trace_execution(logger=logger)
    def build(self, records: List[Dict[str, Any]], modality: str) -> Path:
        """Encode *records* and write a FAISS index to disk.

        Raises
        ------
        IndexBuildError
            When FAISS and the embedding model are both available but the index
            could not be built, or was built empty.  Previously this path wrote a
            zero-byte index file and logged a warning, so a broken index looked
            exactly like an empty corpus at query time.
        """
        index_path = self.output_dir / f"faiss_{modality}.index"
        meta_path = self.output_dir / f"faiss_{modality}_meta.json"

        if not records:
            logger.warning("DenseIndexBuilder: no records for modality=%s", modality)
            meta_path.write_text("[]", encoding="utf-8")
            self.build_stats[modality] = {"records": 0, "ntotal": 0, "degraded": "no_records"}
            return index_path

        texts = [r.get("text", "") or "" for r in records]
        meta = [
            {
                "id": r.get("id", str(i)),
                "document_id": r.get("document_id", ""),
                "region_id": (r.get("metadata") or {}).get("region_id") or r.get("region_id", ""),
                "modality": r.get("modality", modality),
                "metadata": r.get("metadata", {}),
                "text": texts[i][:500],
            }
            for i, r in enumerate(records)
        ]
        meta_json = json.dumps(meta, indent=2, ensure_ascii=False)

        # Optional-dependency probe: absence is a degradation, not a failure.
        try:
            import faiss
        except ImportError:
            logger.warning("DenseIndexBuilder: faiss unavailable - dense retrieval will degrade for %s", modality)
            meta_path.write_text(meta_json, encoding="utf-8")
            index_path.unlink(missing_ok=True)
            self.build_stats[modality] = {"records": len(records), "ntotal": 0, "degraded": "faiss_unavailable"}
            return index_path

        model = self._get_model()
        if model is None:
            logger.warning(
                "DenseIndexBuilder: embedding model unavailable - dense retrieval will degrade for %s", modality
            )
            meta_path.write_text(meta_json, encoding="utf-8")
            index_path.unlink(missing_ok=True)
            self.build_stats[modality] = {"records": len(records), "ntotal": 0, "degraded": "embedder_unavailable"}
            return index_path

        # Real build: any failure from here is a defect, not a degradation.
        try:
            logger.info("DenseIndexBuilder: encoding %d records (modality=%s)", len(texts), modality)
            vectors = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,  # cosine via inner product
            ).astype("float32")

            dim = vectors.shape[1] if hasattr(vectors, "shape") and len(vectors.shape) > 1 else _DIM

            resolved_type = self._resolve_index_type(len(vectors), faiss)
            if resolved_type == "hnsw":
                logger.info(
                    "DenseIndexBuilder: building IndexHNSWFlat (dim=%d, M=%d, efConstruction=%d)",
                    dim,
                    self.hnsw_m,
                    self.hnsw_ef_construction,
                )
                index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = self.hnsw_ef_construction
            else:
                index = faiss.IndexFlatIP(dim)

            index.add(vectors)

            # Fail loudly rather than persisting an index that answers every query
            # with an empty list.
            if index.ntotal != len(vectors):
                raise IndexBuildError(
                    f"FAISS index for modality={modality} holds {index.ntotal} vectors "
                    f"but {len(vectors)} were added"
                )

            faiss.write_index(index, str(index_path))
            meta_path.write_text(meta_json, encoding="utf-8")
            # A rebuild invalidates both caches; without this, search() in the same
            # process would keep answering from the index this call just replaced.
            with self._load_lock:
                self._index_cache.pop(modality, None)
                self._meta_cache.pop(modality, None)
            self.build_stats[modality] = {
                "records": len(records),
                "ntotal": int(index.ntotal),
                "index_type": resolved_type,
                "degraded": None,
            }
            logger.info(
                "DenseIndexBuilder: wrote %s (%d vectors, type=%s)", index_path.name, index.ntotal, resolved_type
            )

        except IndexBuildError:
            meta_path.write_text(meta_json, encoding="utf-8")
            self.build_stats[modality] = {"records": len(records), "ntotal": 0, "degraded": "build_empty"}
            raise
        except Exception as exc:
            logger.error("DenseIndexBuilder: FAISS build failed for modality=%s: %s", modality, exc)
            meta_path.write_text(meta_json, encoding="utf-8")
            self.build_stats[modality] = {"records": len(records), "ntotal": 0, "degraded": "build_error"}
            raise IndexBuildError(f"FAISS build failed for modality={modality}: {exc}") from exc

        return index_path

    def _resolve_index_type(self, n_vectors: int, faiss_module: Any) -> str:
        """Resolve the configured index type against the corpus size.

        An HNSW graph only pays for itself once exhaustive search is the bottleneck.
        Below ``hnsw_min_vectors`` a flat inner-product index is *exact* and faster,
        so requesting "hnsw" on a small corpus resolves to flat rather than paying
        graph build cost for worse recall.
        """
        if self.index_type != "hnsw" or not hasattr(faiss_module, "IndexHNSWFlat"):
            return "flat"
        if n_vectors < self.hnsw_min_vectors:
            logger.info(
                "DenseIndexBuilder: %d vectors is below hnsw_min_vectors=%d — using an exact flat index",
                n_vectors,
                self.hnsw_min_vectors,
            )
            return "flat"
        return "hnsw"

    def ntotal(self, modality: str) -> int:
        """Vectors in the on-disk index for *modality*; 0 when absent or unreadable."""
        index_path = self.output_dir / f"faiss_{modality}.index"
        if not index_path.exists() or index_path.stat().st_size == 0:
            return 0
        try:
            import faiss

            return int(faiss.read_index(str(index_path)).ntotal)
        except Exception:
            return 0

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

            meta = self._meta_cache.get(modality)
            index = self._index_cache.get(modality)
            if meta is None or index is None:
                with self._load_lock:
                    # Re-check inside the lock: another thread may have loaded both
                    # while this one waited.
                    meta = self._meta_cache.get(modality)
                    if meta is None:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        self._meta_cache[modality] = meta
                    index = self._index_cache.get(modality)
                    if index is None:
                        index = faiss.read_index(str(index_path))
                        self._index_cache[modality] = index

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
        """Lazy-load and cache the SentenceTransformer model (thread-safe)."""
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            return self._load_model_locked()

    def _load_model_locked(self) -> Optional[Any]:
        try:
            from sentence_transformers import SentenceTransformer

            # revision pins the weights so a hub-side model update cannot silently
            # change embeddings between two runs of the same code.
            kwargs = {"revision": self.model_revision} if self.model_revision else {}
            self._model = SentenceTransformer(self.model_name, **kwargs)
            logger.info(
                "SentenceTransformer loaded: %s (revision=%s)",
                self.model_name,
                self.model_revision or "hub-default",
            )
        except Exception as exc:
            # Pinning a revision makes the loader resolve that commit against the hub.
            # When the hub is unreachable but the commit is already in the local cache,
            # the right answer is the cached weights, not a degraded index — so retry
            # offline before giving up. Without this, adding a pin turned every
            # network hiccup into a silently empty dense index.
            self._model = self._load_from_cache(exc)
        return self._model

    def _load_from_cache(self, first_error: Exception) -> Optional[Any]:
        """Second attempt at the pinned weights, restricted to the local cache."""
        try:
            from sentence_transformers import SentenceTransformer

            kwargs: Dict[str, Any] = {"local_files_only": True}
            if self.model_revision:
                kwargs["revision"] = self.model_revision
            model = SentenceTransformer(self.model_name, **kwargs)
            logger.warning(
                "Hub unreachable (%s) — loaded %s from the local cache at revision %s",
                first_error,
                self.model_name,
                self.model_revision or "hub-default",
            )
            return model
        except Exception as exc:
            logger.warning(
                "Failed to load SentenceTransformer (%s; cache retry: %s) — dense retrieval will degrade",
                first_error,
                exc,
            )
            return None

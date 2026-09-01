"""sparse_index.py — BM25S sparse retrieval index for document search.

Uses the ``bm25s`` library (pure Python, pip-installable, no Java/Lucene)
to build a BM25 index over text chunks for the text and table retrieval
paths.

Files produced
--------------
* ``{output_dir}/bm25_{modality}/`` — native bm25s serialization directory

Fallback behaviour
------------------
If ``bm25s`` is not installed, the builder writes a ``bm25_{modality}_stub.json``
file so callers always receive consistent (empty) search results rather
than raising ``ImportError``.

Usage
-----
::

    from docureason.tripath.indexing.sparse_index import BM25SIndexBuilder

    builder = BM25SIndexBuilder(output_dir=Path("artifacts/output"))
    builder.build(records, modality="text")
    results = builder.search("quarterly revenue", modality="text", k=10)
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


from docureason.tripath.config import DocuReasonConfig, IndexingConfig


class SparseIndexBuildError(RuntimeError):
    """Raised when a BM25S index could not be built despite ``bm25s`` being importable.

    A missing ``bm25s`` install is a *degradation* (recorded in ``build_stats`` and
    marked on disk with a ``.stub`` file).  Anything else is a defect and must not
    be swallowed into a silently empty index.
    """


class BM25SIndexBuilder:
    """Build and query a BM25S sparse index per modality.

    Parameters
    ----------
    output_dir:
        Directory where ``bm25_{modality}/`` index folders are saved.
    config:
        Optional DocuReasonConfig or IndexingConfig instance.
    """

    def __init__(
        self, output_dir: str | Path, config: Optional[Union[IndexingConfig, DocuReasonConfig]] = None
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
        self.bm25_k1 = idx_cfg.bm25_k1 if idx_cfg else 1.5
        self.bm25_b = idx_cfg.bm25_b if idx_cfg else 0.75
        self._indices: Dict[str, Any] = {}  # in-memory cache after build
        self._meta: Dict[str, List[Dict[str, Any]]] = {}
        #: Guards the lazy search-time load; the parallel modality dispatch calls
        #: search() for text and table concurrently against this one builder.
        self._load_lock = threading.Lock()
        #: Per-modality build observations: {"records", "indexed", "degraded"}.
        self.build_stats: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, records: List[Dict[str, Any]], modality: str) -> Path:
        """Tokenize *records* and persist a BM25S index.

        Parameters
        ----------
        records:
            List of dicts with at minimum ``{"id": str, "text": str}``.
        modality:
            Index name stem (``"text"``, ``"table"``, ``"vision"``).

        Returns
        -------
        Path to the index directory.
        """
        index_dir = self.output_dir / f"bm25_{modality}"
        index_dir.mkdir(parents=True, exist_ok=True)

        meta = [
            {
                "id": r.get("id", str(i)),
                "document_id": r.get("document_id", ""),
                # Carried so retrieval hits join back to manifest.json without a
                # second lookup (audit P1-1).
                "region_id": (r.get("metadata") or {}).get("region_id") or r.get("region_id", ""),
                "modality": r.get("modality", modality),
                "text": (r.get("text", "") or "")[:500],
                "metadata": r.get("metadata", {}),
            }
            for i, r in enumerate(records)
        ]

        texts = [r.get("text", "") or "" for r in records]
        self._meta[modality] = meta

        if not texts:
            logger.warning("BM25SIndexBuilder: no texts for modality=%s", modality)
            (index_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            self.build_stats[modality] = {"records": 0, "indexed": 0, "degraded": "no_records"}
            return index_dir

        try:
            import bm25s

            corpus_tokens = bm25s.tokenize(texts, stopwords="en")
            # k1/b come from IndexingConfig; previously they were read into
            # self.* and never passed, so config changes had no effect.
            retriever = bm25s.BM25(k1=self.bm25_k1, b=self.bm25_b)
            retriever.index(corpus_tokens)
            retriever.save(str(index_dir))
            (index_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            self._indices[modality] = retriever
            (index_dir / ".stub").unlink(missing_ok=True)
            self.build_stats[modality] = {"records": len(texts), "indexed": len(texts), "degraded": None}
            logger.info(
                "BM25SIndexBuilder: indexed %d docs (modality=%s) → %s",
                len(texts),
                modality,
                index_dir,
            )

        except ImportError:
            logger.warning("bm25s not installed — writing stub. Install with: pip install bm25s")
            (index_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            # Mark as stub so search() knows not to try loading.
            (index_dir / ".stub").write_text("no-bm25s", encoding="utf-8")
            self.build_stats[modality] = {"records": len(texts), "indexed": 0, "degraded": "bm25s_unavailable"}

        except Exception as exc:
            # bm25s is present but the build failed: that is a defect. Writing meta
            # and returning would leave an index that answers every query with [].
            logger.error("BM25SIndexBuilder build failed for modality=%s: %s", modality, exc)
            (index_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            self.build_stats[modality] = {"records": len(texts), "indexed": 0, "degraded": "build_error"}
            raise SparseIndexBuildError(f"BM25S build failed for modality={modality}: {exc}") from exc

        return index_dir

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        modality: str,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """BM25 search over the index for *modality*.

        Returns a list of ``{"id", "document_id", "text", "score", "modality"}``
        dicts sorted by descending BM25 score.  Returns empty list on any
        failure.
        """
        index_dir = self.output_dir / f"bm25_{modality}"
        if not index_dir.exists():
            return []

        # Stub check — bm25s was not installed when build() ran.
        if (index_dir / ".stub").exists():
            return []

        meta = self._meta.get(modality)
        if meta is None:
            meta_path = index_dir / "meta.json"
            if not meta_path.exists():
                return []
            with self._load_lock:
                meta = self._meta.get(modality)
                if meta is None:
                    # Cached: this file was re-read from disk on every single query before.
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    self._meta[modality] = meta
        if not meta:
            return []

        # Use cached in-memory index if available.
        retriever = self._indices.get(modality)

        try:
            import bm25s

            if retriever is None:
                with self._load_lock:
                    retriever = self._indices.get(modality)
                    if retriever is None:
                        retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)
                        self._indices[modality] = retriever

            query_tokens = bm25s.tokenize([query], stopwords="en")
            results, scores = retriever.retrieve(query_tokens, k=min(k, len(meta)))
            output = []
            for idx, score in zip(results[0], scores[0]):
                if 0 <= idx < len(meta):
                    entry = dict(meta[idx])
                    entry["score"] = round(float(score), 4)
                    output.append(entry)
            return output

        except Exception as exc:
            logger.warning("BM25SIndexBuilder.search failed: %s", exc)
            # Degrade to simple keyword overlap scoring.
            return self._keyword_fallback(query, meta, k)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_fallback(query: str, meta: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Trivial keyword overlap scorer used when bm25s is absent."""
        terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for entry in meta:
            words = set(re.findall(r"\w+", (entry.get("text") or "").lower()))
            score = len(terms & words) / max(1, len(terms))
            if score > 0:
                scored.append({**entry, "score": round(score, 4)})
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:k]

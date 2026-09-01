"""index_backed.py — shared machinery for index-backed modality retrieval.

Before this change every modality retriever was an in-memory substring scan over
every region of every document: ``O(R)`` per query, no index, and demonstrably
wrong rankings — the two-letter token ``in`` matched *inside* the word
``financial`` and that spurious hit was then multiplied by the 2.5x body field
weight.  Meanwhile ``DenseIndexBuilder.search`` and ``BM25SIndexBuilder.search``
were fully implemented with **zero callers**, so the framework's headline claim
(hybrid dense + sparse multimodal retrieval) was never exercised at query time.

This module holds the pieces the three retrievers share:

* :func:`build_lookups` — region- and chunk-level indexes over the in-memory
  corpus, so an index hit resolves to full untruncated text and to a canonical
  ``region_id`` that joins back to ``manifest.json``.
* :func:`merge_index_hits` — rank-based Reciprocal Rank Fusion of the dense and
  sparse hit lists.  Rank-based because cosine similarity and BM25 scores live on
  different unbounded scales; combining them in score space would need a
  calibration step that does not exist.
* :func:`resolve_top_k` — the per-modality candidate cap.  ``top_k_text`` /
  ``top_k_table`` / ``top_k_vision`` were defined in config with zero consumers,
  so every retriever returned every match and the cross-encoder reranked all of
  them instead of a bounded candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..ingestion.identity import canonical_region_id
from ..ingestion.schema import Chunk, Document, Region

#: Index hits are chunk-level but the merge dedupes to one entry per region, so a
#: k-region result needs more than k chunk hits fetched from each index.
_OVERFETCH_FACTOR = 3
_OVERFETCH_FLOOR = 10

_TOP_K_FIELDS = {
    "text": "top_k_text",
    "table": "top_k_table",
    "vision": "top_k_vision",
}


@dataclass
class CorpusLookups:
    """Region- and chunk-level indexes over an in-memory corpus."""

    #: canonical region_id -> (owning document, region, region index)
    region_by_id: Dict[str, Tuple[Document, Region, int]] = field(default_factory=dict)
    #: chunk id -> chunk, used to recover the full text of the window that matched
    chunk_by_id: Dict[str, Chunk] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.region_by_id)


def build_lookups(
    documents: Sequence[Document],
    region_types: Optional[Iterable[str]] = None,
) -> CorpusLookups:
    """Index *documents* by canonical region id and by chunk id.

    ``region_types`` restricts the region index to one modality's region types so a
    text index hit can never resolve to a table region.
    """
    wanted = frozenset(region_types) if region_types else None
    lookups = CorpusLookups()

    for document in documents:
        for idx, region in enumerate(getattr(document, "regions", []) or []):
            if wanted is not None and region.type not in wanted:
                continue
            lookups.region_by_id[canonical_region_id(document.id, region, idx)] = (document, region, idx)
        for chunk in getattr(document, "chunks", []) or []:
            if chunk.id:
                lookups.chunk_by_id[chunk.id] = chunk

    return lookups


def hit_region_id(hit: Dict[str, Any]) -> str:
    """Best-effort canonical region id for one raw index hit."""
    metadata = hit.get("metadata") or {}
    for value in (hit.get("region_id"), metadata.get("region_id")):
        if value:
            return str(value)
    return str(hit.get("document_id") or hit.get("id") or "")


@dataclass
class MergedHit:
    """One region surviving the dense/sparse merge, resolved against the corpus."""

    region_id: str
    document: Document
    region: Region
    region_index: int
    #: Highest-ranked chunk of this region, when the corpus carries chunks.
    chunk: Optional[Chunk]
    rrf_score: float
    #: Per-run 1-based rank of the best chunk for this region, e.g. {"dense": 2}.
    ranks: Dict[str, int]


def merge_index_hits(
    runs: Dict[str, List[Dict[str, Any]]],
    lookups: CorpusLookups,
    rrf_k: int = 60,
) -> Tuple[List[MergedHit], Dict[str, int]]:
    """RRF-merge per-run index hit lists into region-level candidates.

    Each run is assumed to arrive sorted by descending relevance (both index
    ``search()`` implementations guarantee that).  Within a run only the
    best-ranked chunk of a given region contributes, so a region split into six
    windows cannot occupy six of the fused slots.

    Returns the merged hits (descending score) plus a small stats dict:
    ``{"stale_hits": n, "resolved": n}``.  ``stale_hits`` counts index entries
    whose region is absent from the current corpus, which is the signal that an
    index was built against a different document set.
    """
    scores: Dict[str, float] = {}
    ranks: Dict[str, Dict[str, int]] = {}
    best_chunk_id: Dict[str, str] = {}
    stale = 0

    for run_name, hits in runs.items():
        if not hits:
            continue
        seen: set[str] = set()
        rank = 0
        for hit in hits:
            region_id = hit_region_id(hit)
            if not region_id or region_id in seen:
                continue
            if region_id not in lookups.region_by_id:
                stale += 1
                continue
            seen.add(region_id)
            rank += 1
            scores[region_id] = scores.get(region_id, 0.0) + 1.0 / (rrf_k + rank)
            ranks.setdefault(region_id, {})[run_name] = rank
            best_chunk_id.setdefault(region_id, str(hit.get("id") or ""))

    merged: List[MergedHit] = []
    for region_id, score in scores.items():
        document, region, region_index = lookups.region_by_id[region_id]
        merged.append(
            MergedHit(
                region_id=region_id,
                document=document,
                region=region,
                region_index=region_index,
                chunk=lookups.chunk_by_id.get(best_chunk_id.get(region_id, "")),
                rrf_score=score,
                ranks=ranks.get(region_id, {}),
            )
        )

    merged.sort(key=lambda hit: hit.rrf_score, reverse=True)
    return merged, {"stale_hits": stale, "resolved": len(merged)}


def resolve_top_k(config: Any, modality: str, explicit: Optional[int] = None) -> int:
    """Per-modality candidate cap, in precedence order: explicit → config → 20.

    ``explicit=0`` (or a negative config value) means "no cap" and is returned as
    ``0``; callers treat any non-positive value as unbounded.
    """
    if explicit is not None:
        return max(0, int(explicit))

    retrieval = getattr(config, "retrieval", None)
    field_name = _TOP_K_FIELDS.get(modality, "top_k_text")
    value = getattr(retrieval, field_name, None) if retrieval is not None else None
    if value is None:
        return 20
    return max(0, int(value))


def fetch_k(top_k: int) -> int:
    """How many chunk-level hits to request from an index for a *top_k* region result."""
    if top_k <= 0:
        # Unbounded cap: still bound the index read so a query cannot pull the
        # whole corpus into memory.
        return 200
    return max(top_k * _OVERFETCH_FACTOR, top_k + _OVERFETCH_FLOOR)


def apply_cap(items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Truncate *items* to *top_k*, treating a non-positive cap as unbounded."""
    return items[:top_k] if top_k > 0 else items


class IndexBackedRetriever:
    """Template for a modality retriever: query the indexes, fall back to a scan.

    Subclasses declare :attr:`modality`, :attr:`region_types` and
    :attr:`uses_sparse`, and implement :meth:`payload_from_hit` and
    :meth:`scan`.  Everything else - cap resolution, the dense/sparse merge, the
    fallback decision and the statistics that feed degradation reporting - lives
    here so the three retrievers cannot drift apart.
    """

    #: Index name and the value stamped on every result.
    modality: str = "text"
    #: Region types this modality owns, so a hit can never resolve across modalities.
    region_types: frozenset = frozenset()
    #: False for modalities the pipeline does not build a BM25S index for.
    uses_sparse: bool = True

    def __init__(
        self,
        dense: Optional[Any] = None,
        sparse: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> None:
        self.dense = dense
        self.sparse = sparse if self.uses_sparse else None
        self.config = config
        self.last_stats: Dict[str, Any] = {"mode": "uninitialized"}

    # -- configuration ------------------------------------------------------

    def index_enabled(self) -> bool:
        if self.dense is None and self.sparse is None:
            return False
        retrieval = getattr(self.config, "retrieval", None)
        return bool(getattr(retrieval, "enable_index_retrieval", True))

    def rrf_k(self) -> int:
        retrieval = getattr(self.config, "retrieval", None)
        return int(getattr(retrieval, "rrf_k", 60) or 60)

    # -- template -----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        cap = resolve_top_k(self.config, self.modality, top_k)
        if not str(query).strip():
            self.last_stats = {"mode": "empty_query"}
            return []

        index_stats: Dict[str, Any] = {}
        if self.index_enabled():
            results, index_stats = self._retrieve_from_index(query, documents, cap)
            if results:
                self.last_stats = {"mode": "hybrid_index", **index_stats}
                return results

        results = self.scan(query, documents)
        # The index attempt's statistics are carried through: a stale-index signal
        # must not be lost just because the fallback produced the answer.
        self.last_stats = {
            "mode": "substring_scan",
            "candidates": len(results),
            "index_available": self.index_enabled(),
            **index_stats,
        }
        return apply_cap(results, cap)

    def _retrieve_from_index(
        self,
        query: str,
        documents: Sequence[Document],
        cap: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        lookups = build_lookups(documents, self.region_types)
        if not lookups:
            return [], {}

        k = fetch_k(cap)
        runs: Dict[str, List[Dict[str, Any]]] = {}
        if self.dense is not None:
            runs["dense"] = self.dense.search(query, self.modality, k=k) or []
        if self.sparse is not None:
            runs["sparse"] = self.sparse.search(query, self.modality, k=k) or []

        merged, merge_stats = merge_index_hits(runs, lookups, rrf_k=self.rrf_k())
        stats = {
            "dense_hits": len(runs.get("dense", [])),
            "sparse_hits": len(runs.get("sparse", [])),
            **merge_stats,
        }
        if not merged:
            return [], stats
        return apply_cap([self.payload_from_hit(hit) for hit in merged], cap), stats

    # -- subclass hooks -----------------------------------------------------

    def payload_from_hit(self, hit: MergedHit) -> Dict[str, Any]:
        raise NotImplementedError

    def scan(self, query: str, documents: Sequence[Document]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    # -- helpers for subclasses --------------------------------------------

    @staticmethod
    def child_text(hit: MergedHit) -> str:
        """Text of the chunk that matched, or the whole region when unchunked."""
        return hit.chunk.text if hit.chunk is not None else hit.region.text

    @staticmethod
    def base_payload(hit: MergedHit, modality: str, weight: float) -> Dict[str, Any]:
        return {
            "document_id": hit.document.id,
            "region_id": hit.region_id,
            "chunk_id": hit.chunk.id if hit.chunk is not None else None,
            # RRF scores are strictly positive, so a multiplicative field weight is
            # monotone here (unlike the signed cross-encoder logits).
            "score": round(hit.rrf_score * weight * 100.0, 4),
            "rrf_score": round(hit.rrf_score, 6),
            "modality": modality,
            "field_weight": weight,
            "retrieval_mode": "hybrid_index",
            "dense_rank": hit.ranks.get("dense"),
            "sparse_rank": hit.ranks.get("sparse"),
        }

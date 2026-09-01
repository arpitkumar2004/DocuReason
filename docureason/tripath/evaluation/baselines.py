"""baselines.py — the naive RAG baseline every reported number needs.

The report and the docs quote deltas against "naive RAG" without a naive RAG existing
anywhere in the repository, so nothing the tri-path system achieved could be
attributed to the tri-path architecture rather than to the corpus, the embedding
model, or the generator.  A baseline is the cheapest thing in evaluation and the one
that makes every other number mean something.

This is deliberately the *textbook* pipeline, so the comparison isolates retrieval
architecture and nothing else:

* recursive character splitting on paragraph / sentence / word boundaries - no layout
  parsing, no section awareness, no parent-child hierarchy;
* the same embedding model at the same pinned revision as the full system;
* one flat exact inner-product index over all chunks, no per-modality split, no BM25,
  no fusion, no reranking, no SQL path;
* top-k by cosine similarity, k=5;
* **the same generator**, called the same way.

Deviating on the generator or the embedding model would make the delta unreadable:
the point is that only the retrieval architecture differs.

Region attribution
------------------
A naive splitter cuts across the region boundaries that gold labels are written
against, so each chunk records every region it overlaps (``region_ids``) and names
the region contributing the most characters as its primary ``region_id``.  Scoring a
chunk as a miss because its dominant region was the neighbour of the gold one would
measure the label format, not the retriever.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from docureason.tripath.config import DocuReasonConfig
from docureason.tripath.ingestion.identity import canonical_region_id
from docureason.tripath.model_pins import resolve_revision
from docureason.tripath.utils import StageTimer, get_logger

logger = get_logger(__name__)

#: Split points tried in order, coarsest first - the standard recursive-splitter
#: ladder. The last is a hard character cut, which is why the splitter always makes
#: progress even on text with no whitespace at all.
_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


@dataclass
class NaiveChunk:
    """One fixed-size window of a document, with the regions it overlaps."""

    chunk_id: str
    document_id: str
    text: str
    region_id: str
    region_ids: List[str] = field(default_factory=list)

    def as_payload(self, score: float, rank: int, mode: str) -> Dict[str, Any]:
        return {
            "id": self.chunk_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "region_id": self.region_id,
            "region_ids": list(self.region_ids),
            "text": self.text,
            "score": round(float(score), 4),
            "rank": rank,
            "modality": "text",
            "retrieval_mode": mode,
        }


def recursive_split(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
    """``(start, end)`` character spans for *text*, recursively split.

    Spans rather than strings: the caller needs the offsets to work out which regions
    each chunk overlaps, and slicing twice would be the only alternative.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    stride = max(1, chunk_size - max(0, overlap))
    spans: List[Tuple[int, int]] = []
    position = 0
    length = len(text)

    while position < length:
        end = min(position + chunk_size, length)
        if end < length:
            end = _best_break(text, position, end)
        spans.append((position, end))
        if end >= length:
            break
        position = max(position + stride, position + 1) if end - position >= stride else end
    return spans


def _best_break(text: str, start: int, hard_end: int) -> int:
    """The latest natural boundary at or before *hard_end*, else *hard_end*."""
    window = text[start:hard_end]
    for separator in _SEPARATORS:
        if not separator:
            break
        cut = window.rfind(separator)
        # Refuse a break in the first quarter of the window: it would emit a sliver
        # and push the real content into the next chunk.
        if cut > len(window) // 4:
            return start + cut + len(separator)
    return hard_end


class NaiveRAGBaseline:
    """Recursive-split + flat-index + top-k retrieval, sharing the full system's generator."""

    def __init__(
        self,
        config: Optional[DocuReasonConfig] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        generator: Optional[Any] = None,
    ) -> None:
        self.config = config or DocuReasonConfig.load_preset("balanced")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.chunks: List[NaiveChunk] = []
        self.degradation: List[str] = []
        self._model: Any = None
        self._index: Any = None
        self._generator = generator

        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model_revision = resolve_revision(self.model_name)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, documents: Sequence[Any]) -> Dict[str, Any]:
        """Split every document, embed the chunks, and build one flat index."""
        self.chunks = []
        self.degradation = []
        for document in documents:
            self.chunks.extend(self._split_document(document))

        stats: Dict[str, Any] = {
            "documents": len(documents),
            "chunks": len(self.chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "embedding_model": self.model_name,
            "embedding_revision": self.model_revision or "hub-default",
            "index_type": "flat_inner_product",
        }

        if not self.chunks:
            self.degradation.append("baseline:empty_corpus")
            stats["degraded"] = "empty_corpus"
            return stats

        vectors = self._encode([chunk.text for chunk in self.chunks])
        if vectors is None:
            self.degradation.append("baseline:embedder_unavailable")
            stats["degraded"] = "embedder_unavailable"
            stats["index_type"] = "keyword_overlap_fallback"
            return stats

        try:
            import faiss

            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            if index.ntotal != len(self.chunks):
                raise RuntimeError(f"index holds {index.ntotal} of {len(self.chunks)} chunks")
            self._index = index
            stats["ntotal"] = int(index.ntotal)
        except Exception as exc:
            # Same policy as the full system's index builders: a missing optional
            # dependency degrades and says so, rather than answering every query with
            # an empty list and looking like a corpus with no relevant documents.
            logger.warning("NaiveRAGBaseline: flat index unavailable (%s) — using keyword overlap", exc)
            self.degradation.append("baseline:faiss_unavailable")
            stats["degraded"] = "faiss_unavailable"
            stats["index_type"] = "keyword_overlap_fallback"

        return stats

    def _split_document(self, document: Any) -> List[NaiveChunk]:
        """Concatenate the document's regions, then split without regard for them."""
        pieces: List[str] = []
        offsets: List[Tuple[int, int, str]] = []  # (start, end, region_id)
        cursor = 0

        for index, region in enumerate(getattr(document, "regions", []) or []):
            text = str(getattr(region, "text", "") or "")
            if not text.strip():
                continue
            region_id = canonical_region_id(document.id, region, index)
            pieces.append(text)
            offsets.append((cursor, cursor + len(text), region_id))
            cursor += len(text) + 2  # the "\n\n" join below

        if not pieces:
            return []

        full_text = "\n\n".join(pieces)
        chunks: List[NaiveChunk] = []
        for number, (start, end) in enumerate(recursive_split(full_text, self.chunk_size, self.chunk_overlap)):
            body = full_text[start:end].strip()
            if not body:
                continue
            overlapped = _overlapping_regions(start, end, offsets)
            chunks.append(
                NaiveChunk(
                    chunk_id=f"{document.id}-naive-{number}",
                    document_id=str(document.id),
                    text=body,
                    region_id=overlapped[0] if overlapped else "",
                    region_ids=overlapped,
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, text: str) -> Dict[str, Any]:
        """Retrieve top-k chunks and generate an answer, in the same response shape.

        The shape matches :meth:`QueryService.query` closely enough for the evaluation
        harness to score both systems through one code path - which is the point, since
        a baseline scored by different code is not a baseline.
        """
        timer = StageTimer()
        with timer.stage("dense_search"):
            results = self._search(text)
        with timer.stage("generate"):
            generation = self._generate(text, results)

        degradation = list(self.degradation)
        engine = generation.get("engine", "unknown")
        if engine not in ("local_slm_deepseek_r1", "cloud_gemini_api"):
            degradation.append(f"generation:{engine}")

        return {
            "query": text,
            "system": "naive_rag_baseline",
            "results": results,
            "retrieved_evidence": results[: self.top_k],
            "answer": generation.get("answer", ""),
            "generation": generation,
            "generation_engine": engine,
            "citations": generation.get("citations", []),
            "retrieval_stats": {
                "mode": "flat_index" if self._index is not None else "keyword_overlap",
                "candidates": len(self.chunks),
                "returned": len(results),
            },
            "latency_breakdown": timer.as_dict(),
            "degradation": degradation,
            "cached": False,
        }

    def _search(self, text: str) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        if self._index is not None:
            query_vector = self._encode([text])
            if query_vector is not None:
                try:
                    import faiss  # noqa: F401

                    scores, indices = self._index.search(query_vector, min(self.top_k, len(self.chunks)))
                    payloads = []
                    for rank, (score, position) in enumerate(zip(scores[0], indices[0]), start=1):
                        if position < 0 or position >= len(self.chunks):
                            continue
                        payloads.append(self.chunks[position].as_payload(score, rank, "flat_index"))
                    return payloads
                except Exception as exc:
                    logger.warning("NaiveRAGBaseline: flat search failed (%s) — using keyword overlap", exc)

        return self._keyword_search(text)

    def _keyword_search(self, text: str) -> List[Dict[str, Any]]:
        """Term-overlap ranking, used only when the model or index is unavailable."""
        terms = set(re.findall(r"\w+", text.lower()))
        if not terms:
            return []
        scored: List[Tuple[float, NaiveChunk]] = []
        for chunk in self.chunks:
            words = set(re.findall(r"\w+", chunk.text.lower()))
            overlap = len(terms & words) / max(1, len(terms))
            if overlap > 0:
                scored.append((overlap, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            chunk.as_payload(score, rank, "keyword_overlap")
            for rank, (score, chunk) in enumerate(scored[: self.top_k], start=1)
        ]

    def _generate(self, text: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        generator = self._get_generator()
        if generator is None:
            return {"answer": "", "engine": "unavailable", "citations": []}
        # No SQL payload: the baseline has no SQL path, and passing an empty one is
        # what "the same generator, less input" means.
        return generator.generate(text, results, {"executed": False, "sql_results": []})

    def _get_generator(self) -> Optional[Any]:
        if self._generator is not None:
            return self._generator
        try:
            from docureason.tripath.generation.generate import GenerationModule

            self._generator = GenerationModule(config=self.config)
        except Exception as exc:
            logger.warning("NaiveRAGBaseline: generator unavailable: %s", exc)
            self._generator = None
        return self._generator

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> Optional[Any]:
        model = self._get_model()
        if model is None:
            return None
        try:
            started = time.perf_counter()
            vectors = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            logger.debug("NaiveRAGBaseline: encoded %d text(s) in %.3fs", len(texts), time.perf_counter() - started)
            return vectors.astype("float32")
        except Exception as exc:
            logger.warning("NaiveRAGBaseline: encoding failed: %s", exc)
            return None

    def _get_model(self) -> Optional[Any]:
        if self._model is not None:
            return self._model
        for kwargs in ({"revision": self.model_revision} if self.model_revision else {}, {"local_files_only": True}):
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, **kwargs)
                return self._model
            except Exception as exc:
                logger.debug("NaiveRAGBaseline: model load attempt failed (%s)", exc)
        logger.warning("NaiveRAGBaseline: embedding model unavailable — falling back to keyword overlap")
        return None


def _overlapping_regions(start: int, end: int, offsets: List[Tuple[int, int, str]]) -> List[str]:
    """Region ids overlapping ``[start, end)``, the largest contributor first."""
    overlaps: List[Tuple[int, str]] = []
    for region_start, region_end, region_id in offsets:
        shared = min(end, region_end) - max(start, region_start)
        if shared > 0:
            overlaps.append((shared, region_id))
    overlaps.sort(key=lambda pair: pair[0], reverse=True)
    return [region_id for _, region_id in overlaps]


__all__ = ["NaiveChunk", "NaiveRAGBaseline", "recursive_split"]

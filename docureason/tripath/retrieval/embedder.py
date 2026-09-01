from __future__ import annotations

import hashlib
import warnings
from typing import List

from ..ingestion.schema import Chunk


class ChunkEmbedder:
    """DEPRECATED — produces deterministic hash vectors, not semantic embeddings.

    These vectors are variable-length (one dimension per unique token, sorted) and
    carry no semantic signal, so they are unusable for similarity search.  The class
    was previously invoked on the entire corpus for every query in
    ``QueryService.query`` and the result was returned in the response and never
    read — ``O(corpus)`` wasted work plus an unbounded payload.

    Use :class:`docureason.tripath.indexing.dense_index.DenseIndexBuilder` for real
    embeddings.  Retained only so external imports do not break; scheduled for
    removal in the next minor release.
    """

    def __init__(self) -> None:
        warnings.warn(
            "ChunkEmbedder produces non-semantic hash vectors and is deprecated; "
            "use DenseIndexBuilder for embeddings.",
            DeprecationWarning,
            stacklevel=2,
        )

    def embed(self, chunks: List[Chunk]) -> List[dict]:
        embeddings: List[dict] = []
        for chunk in chunks:
            token_vector = self._token_vector(chunk.text)
            embeddings.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "modality": chunk.modality,
                    "vector": token_vector,
                    "text": chunk.text,
                }
            )
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        return self._token_vector(query)

    def _token_vector(self, text: str) -> List[float]:
        tokens = [token.lower() for token in text.split() if token]
        vector: List[float] = []
        for token in sorted(set(tokens)):
            vector.append(self._hash_weight(token))
        return vector

    def _hash_weight(self, token: str) -> float:
        return round(int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFF, 4)

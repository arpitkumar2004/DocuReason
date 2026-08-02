from __future__ import annotations

import hashlib
from typing import List

from ..ingestion.schema import Chunk


class ChunkEmbedder:
    """Create lightweight deterministic embeddings for chunks."""

    def embed(self, chunks: List[Chunk]) -> List[dict]:
        embeddings: List[dict] = []
        for chunk in chunks:
            token_vector = self._token_vector(chunk.text)
            embeddings.append({
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "modality": chunk.modality,
                "vector": token_vector,
                "text": chunk.text,
            })
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

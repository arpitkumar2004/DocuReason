"""Index round-trip: build a real FAISS/BM25S index, then query it through the retrievers.

Covers three things that had no test at all:

* ``DenseIndexBuilder.search`` and ``BM25SIndexBuilder.search`` are reachable from
  the retrievers (they had zero callers anywhere in the repository);
* a failed build **raises** instead of persisting a zero-byte index that answers
  every query with ``[]``;
* the ``top_k_{modality}`` cap is applied to what the retrievers return.

The embedding model is stubbed with a deterministic bag-of-words hash so the test
is hermetic - no HuggingFace download - while still exercising real FAISS.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from docureason.tripath.config import DocuReasonConfig
from docureason.tripath.indexing.dense_index import DenseIndexBuilder, IndexBuildError
from docureason.tripath.indexing.sparse_index import BM25SIndexBuilder
from docureason.tripath.ingestion.schema import Chunk, Document, Region
from docureason.tripath.retrieval.text_retrieval import TextRetrieval

pytestmark = pytest.mark.unit

faiss = pytest.importorskip("faiss", reason="faiss-cpu is a declared dependency")
bm25s = pytest.importorskip("bm25s", reason="bm25s is a declared dependency")

_DIM = 64


class StubEncoder:
    """Deterministic hashed bag-of-words embedder with unit-normalized output."""

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        vectors = np.zeros((len(texts), _DIM), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", str(text).lower()):
                vectors[row, hash(token) % _DIM] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1.0, norms)
        return vectors


def _corpus() -> list[Document]:
    bodies = {
        0: "Quarterly revenue rose to twelve million dollars driven by enterprise licences.",
        1: "Operating margin contracted by two hundred basis points on hiring costs.",
        2: "The board approved a share repurchase programme of fifty million dollars.",
    }
    regions, chunks = [], []
    for idx, text in bodies.items():
        region_id = f"doc-fixture-region-{idx}"
        regions.append(Region(type="body", text=text, metadata={"region_id": region_id}))
        chunks.append(
            Chunk(
                id=f"{region_id}-chunk-0",
                document_id="doc-fixture",
                region_id=region_id,
                modality="text",
                text=text,
                metadata={"region_id": region_id, "parent_text": text + " " + text},
            )
        )
    return [Document(id="doc-fixture", source="fixture.txt", title="Fixture", regions=regions, chunks=chunks)]


def _records(documents):
    return [
        {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "modality": chunk.modality,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for document in documents
        for chunk in document.chunks
    ]


@pytest.fixture
def built(tmp_path):
    documents = _corpus()
    records = _records(documents)

    dense = DenseIndexBuilder(output_dir=tmp_path, index_type="flat")
    dense._model = StubEncoder()
    dense.build(records, modality="text")

    sparse = BM25SIndexBuilder(output_dir=tmp_path)
    sparse.build(records, modality="text")
    return documents, dense, sparse


def test_dense_build_reports_ntotal_and_writes_a_non_empty_index(built, tmp_path) -> None:
    _, dense, _ = built
    # A successful build also records the index type actually resolved, which is not
    # necessarily the requested one (HNSW downgrades to flat below hnsw_min_vectors).
    assert dense.build_stats["text"] == {
        "records": 3,
        "ntotal": 3,
        "index_type": "flat",
        "degraded": None,
    }
    assert dense.ntotal("text") == 3
    assert (tmp_path / "faiss_text.index").stat().st_size > 0


def test_sparse_build_reports_indexed_count(built) -> None:
    _, _, sparse = built
    assert sparse.build_stats["text"] == {"records": 3, "indexed": 3, "degraded": None}


def test_dense_search_finds_the_right_region(built) -> None:
    _, dense, _ = built
    hits = dense.search("revenue rose to twelve million", "text", k=3)
    assert hits, "dense search returned nothing from a populated index"
    assert hits[0]["region_id"] == "doc-fixture-region-0"


def test_sparse_search_finds_the_right_region(built) -> None:
    _, _, sparse = built
    hits = sparse.search("operating margin basis points", "text", k=3)
    assert hits
    assert hits[0]["region_id"] == "doc-fixture-region-1"


def test_retriever_uses_the_index_and_reports_the_mode(built) -> None:
    documents, dense, sparse = built
    retriever = TextRetrieval(dense=dense, sparse=sparse, config=DocuReasonConfig.load_preset("balanced"))
    results = retriever.retrieve("share repurchase programme", documents)

    assert retriever.last_stats["mode"] == "hybrid_index"
    assert retriever.last_stats["stale_hits"] == 0
    assert results[0]["region_id"] == "doc-fixture-region-2"
    assert results[0]["retrieval_mode"] == "hybrid_index"
    # Both runs contributed a rank for the top hit.
    assert results[0]["dense_rank"] is not None
    assert results[0]["sparse_rank"] is not None


def test_retriever_honours_the_top_k_cap(built) -> None:
    documents, dense, sparse = built
    config = DocuReasonConfig.load_preset("balanced")
    config.retrieval.top_k_text = 2
    retriever = TextRetrieval(dense=dense, sparse=sparse, config=config)
    assert len(retriever.retrieve("million dollars revenue margin", documents)) == 2


def test_retriever_falls_back_to_scan_when_disabled(built) -> None:
    documents, dense, sparse = built
    config = DocuReasonConfig.load_preset("balanced")
    config.retrieval.enable_index_retrieval = False
    retriever = TextRetrieval(dense=dense, sparse=sparse, config=config)
    results = retriever.retrieve("revenue", documents)

    assert retriever.last_stats["mode"] == "substring_scan"
    assert all(item["retrieval_mode"] == "substring_scan" for item in results)


def test_stale_index_hits_are_reported_not_silently_dropped(built) -> None:
    """An index built against a different corpus must surface as ``stale_hits``."""
    _, dense, sparse = built
    other_corpus = [Document(id="doc-other", source="other.txt", title="Other", regions=[], chunks=[])]
    other_corpus[0].regions.append(
        Region(type="body", text="unrelated content", metadata={"region_id": "doc-other-region-0"})
    )
    retriever = TextRetrieval(dense=dense, sparse=sparse, config=DocuReasonConfig.load_preset("balanced"))
    retriever.retrieve("revenue", other_corpus)
    assert retriever.last_stats["stale_hits"] > 0


def test_dense_build_raises_instead_of_writing_a_stub(tmp_path) -> None:
    """A build failure with the dependency present must raise, not persist an empty index."""

    class BrokenEncoder:
        def encode(self, *args, **kwargs):
            raise RuntimeError("encoder exploded")

    dense = DenseIndexBuilder(output_dir=tmp_path, index_type="flat")
    dense._model = BrokenEncoder()

    with pytest.raises(IndexBuildError):
        dense.build(_records(_corpus()), modality="text")

    assert dense.build_stats["text"]["degraded"] == "build_error"
    assert not (tmp_path / "faiss_text.index").exists()
    assert dense.ntotal("text") == 0

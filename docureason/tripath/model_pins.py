"""model_pins.py — the one place that says which model weights this framework runs.

Every model was loaded by name only (``SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")``),
which resolves to whatever the HuggingFace hub currently serves for that repo's
default branch.  A hub-side update therefore changes every vector in the index, every
retrieval score measured against it and every reranked ordering — with no version
signal in the artifacts, the logs or the metrics.  Two runs a month apart were not
comparable and nothing said so.

Pins are keyed by **model name**, not stored as a single scalar, because the model
name is itself configurable: ``indexing.domain = "financial"`` swaps the embedding
model, and a scalar pin would then hand one model's commit to a different model.  A
name-keyed registry either has the right commit for the resolved model or has none.

Only commits that have been *observed* resolving for the model in question are
listed.  A model with no entry loads at the hub default, which
:func:`resolve_revision` reports as ``None`` so callers can record
``"hub-default"`` in their provenance rather than imply a pin that does not exist.

To add a pin, resolve the commit from a machine that has the weights::

    python -c "from huggingface_hub import scan_cache_dir; \
        print([(r.repo_id, x.commit_hash, sorted(x.refs)) for r in scan_cache_dir().repos for x in r.revisions])"
"""

from __future__ import annotations

from typing import Dict, Optional

#: model repo id -> pinned commit SHA.  Resolved from the ``main`` ref of each repo.
PINNED_REVISIONS: Dict[str, str] = {
    # Dense embedding models
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "sentence-transformers/all-mpnet-base-v2": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    "sentence-transformers/paraphrase-MiniLM-L6-v2": "c9a2bfebc254878aee8c3aca9e6844d5bbb102d1",
    # Cross-encoder reranker
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    # Figure captioning
    "openai/clip-vit-base-patch32": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
}


def resolve_revision(model_name: Optional[str], explicit: Optional[str] = None) -> Optional[str]:
    """Commit to load *model_name* at: an explicit config pin, else the registry, else None.

    An explicit value always wins, so a caller can pin a model this registry has
    never seen (or deliberately float one) from configuration alone.
    """
    if explicit:
        return str(explicit)
    if not model_name:
        return None
    return PINNED_REVISIONS.get(str(model_name))

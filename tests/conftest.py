"""conftest.py — shared pytest configuration.

Two problems this addresses.

**Non-hermetic suite.** Twelve of the test modules construct real pipeline
components (``DocuReasonPipeline``, ``QueryService``, ``Ranker``,
``DenseIndexBuilder``, ``HybridRetriever``), which trigger Docling conversion,
HuggingFace downloads and FAISS builds.  ``pytest`` is a blocking CI gate, so a hub
outage turned into a red build with no code change, and a local run was slow enough
that developers skipped it.

**No marker split.** Every test was in one bucket.  Tests now carry a marker:
``unit`` for hermetic tests that must run in seconds, ``integration`` for
model-dependent ones.  A module opts in with ``pytestmark = pytest.mark.unit``;
anything unmarked is treated as ``integration``, so the fast gate can never
silently absorb a slow test.

Run the fast gate with ``pytest -m unit`` and the full suite with ``pytest``.
"""

from __future__ import annotations

import os

import pytest

#: transformers probes TensorFlow at import time unless told not to.  On a
#: torch-only install (which is what this project declares) that probe can fail
#: outright and take sentence-transformers down with it, which shows up as a
#: silently degraded dense index rather than an error.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Default every unmarked test to ``integration``."""
    for item in items:
        if not any(item.iter_markers(name="unit")) and not any(item.iter_markers(name="integration")):
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def offline_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every model boundary onto its deterministic fallback.

    Both switches are pre-existing code paths; this fixture just makes them easy to
    request from a test that wants the fallback behaviour rather than the model.
    """
    monkeypatch.setenv("DOCUREASON_DISABLE_SLM", "1")
    monkeypatch.setenv("DOCUREASON_DISABLE_DOCLING", "1")
    monkeypatch.setenv("DOCUREASON_GEN_BACKEND", "template")

"""test_serving_concurrency.py — the corpus is ingested once, not once per request.

Audit §10 blind spot #10 (P1-11).  Every route in ``serving/main.py`` is a
synchronous ``def``, so Starlette runs it in a thread pool.  The service cache was a
bare module-level dict read and written from that pool with no lock, so *N*
concurrent first requests each found it empty and each ran a full ingestion: *N*
Docling conversions, *N* embedding model loads, *N* FAISS builds, and *N* writers
racing on the same artifact directory — from a single burst of traffic.

These tests pin the double-checked lock that fixes it.  ``QueryService`` is replaced
by a counting stub, so what is measured is the caching logic rather than a real
ingestion; the stub sleeps to keep the race window open long enough that an unlocked
implementation reliably fails here.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List

import pytest

from docureason.tripath.serving import main as serving_main

pytestmark = pytest.mark.unit

#: Enough concurrent callers to lose a lock-free double-check reliably.
CALLERS = 20


class CountingService:
    """Stand-in for ``QueryService`` that records every construction."""

    constructions = 0
    _lock = threading.Lock()

    def __init__(self, input_dir: Path, output_dir: Path, verbose: bool = False, **_: Any) -> None:
        with CountingService._lock:
            CountingService.constructions += 1
        # A real QueryService spends seconds here (Docling, embeddings, FAISS).
        time.sleep(0.05)
        self.input_dir = input_dir
        self.output_dir = output_dir

    @classmethod
    def reset(cls) -> None:
        cls.constructions = 0


@pytest.fixture
def counting_services(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[CountingService]]:
    """Swap in the counting stub and give each test an empty service cache."""
    CountingService.reset()
    monkeypatch.setattr(serving_main, "QueryService", CountingService)
    serving_main._SERVICES.clear()
    yield CountingService
    serving_main._SERVICES.clear()
    CountingService.reset()


def _call_concurrently(fn, count: int) -> List[Any]:
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(fn, i) for i in range(count)]
        return [future.result(timeout=30) for future in futures]


def test_concurrent_first_requests_ingest_exactly_once(counting_services) -> None:
    services = _call_concurrently(lambda _: serving_main.get_query_service(), CALLERS)

    assert counting_services.constructions == 1, (
        f"{CALLERS} concurrent first requests triggered {counting_services.constructions} ingestions"
    )
    assert len({id(service) for service in services}) == 1, "callers received different service instances"


def test_a_second_burst_reuses_the_cached_service(counting_services) -> None:
    serving_main.get_query_service()
    _call_concurrently(lambda _: serving_main.get_query_service(), CALLERS)

    assert counting_services.constructions == 1


def test_distinct_corpora_get_distinct_services(counting_services) -> None:
    def call(index: int) -> Any:
        # Two output dirs, each requested by half the callers.
        return serving_main.get_query_service(output_dir=f"artifacts/concurrency_{index % 2}")

    services = _call_concurrently(call, CALLERS)

    assert counting_services.constructions == 2
    assert len({id(service) for service in services}) == 2


def test_equivalent_paths_share_one_service(counting_services) -> None:
    """The cache is keyed on the *resolved* path, so these are one corpus, not three."""
    for candidate in ("samples", "./samples", "samples/../samples"):
        serving_main.get_query_service(input_dir=candidate)

    assert counting_services.constructions == 1


def test_a_rejected_path_never_constructs_a_service(counting_services) -> None:
    with pytest.raises(serving_main.CorpusPathError):
        serving_main.get_query_service(input_dir="../..")

    assert counting_services.constructions == 0
    assert serving_main._SERVICES == {}

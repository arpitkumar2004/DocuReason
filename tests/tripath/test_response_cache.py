"""test_response_cache.py — the response cache, the stage timer, and override isolation.

Three things that must hold together for the serving layer to be measurable:

* a cached response is marked as cached and does **not** replay the original run's
  latency, or every reported percentile drifts toward the cache's;
* the cache key covers the effective config, so an ablation switch cannot be served a
  response computed with the opposite setting;
* per-query overrides do not mutate the service, which they used to.
"""

from __future__ import annotations

import time

import pytest

from docureason.tripath.config import DocuReasonConfig
from docureason.tripath.serving.query_service import QueryService
from docureason.tripath.utils import StageTimer

pytestmark = pytest.mark.unit


def _bare_service(cache_size: int = 3) -> QueryService:
    """A QueryService with only its cache wired.

    Constructed without ``__init__`` on purpose: the real constructor ingests a corpus
    and builds three models, none of which the cache depends on. The alternative is a
    test that takes a minute to assert an eviction.
    """
    service = object.__new__(QueryService)
    service.config = DocuReasonConfig.load_preset("balanced")
    service.config.serving.response_cache_size = cache_size
    import threading
    from collections import OrderedDict

    service._cache = OrderedDict()
    service._cache_lock = threading.Lock()
    service.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
    return service


def _response(answer: str = "a") -> dict:
    return {
        "query": "q",
        "answer": answer,
        "results": [{"region_id": "r1", "text": "evidence"}],
        "latency_breakdown": {"stages_ms": {"rerank": 40.0}, "total_ms": 900.0},
        "degradation": [],
        "cached": False,
    }


def test_a_miss_returns_nothing_and_is_counted() -> None:
    service = _bare_service()
    assert service._cache_get("k", StageTimer()) is None
    assert service.cache_stats == {"hits": 0, "misses": 1, "evictions": 0}


def test_a_hit_is_marked_and_does_not_replay_the_original_latency() -> None:
    service = _bare_service()
    service._cache_put("k", _response(), service.config)

    hit = service._cache_get("k", StageTimer())
    assert hit["cached"] is True
    assert hit["latency_breakdown"]["total_ms"] < 900.0, "a cache lookup is not a pipeline run"
    assert hit["latency_breakdown_first_run"]["total_ms"] == 900.0
    assert service.cache_stats["hits"] == 1


def test_a_hit_is_a_copy_so_callers_cannot_corrupt_the_cache() -> None:
    """The eval harness stamps metrics onto responses; that must not leak across calls."""
    service = _bare_service()
    service._cache_put("k", _response(), service.config)

    first = service._cache_get("k", StageTimer())
    first["results"][0]["text"] = "mutated"
    second = service._cache_get("k", StageTimer())

    assert second["results"][0]["text"] == "evidence"


def test_the_cache_evicts_least_recently_used() -> None:
    service = _bare_service(cache_size=2)
    service._cache_put("a", _response("a"), service.config)
    service._cache_put("b", _response("b"), service.config)
    service._cache_get("a", StageTimer())  # refresh a
    service._cache_put("c", _response("c"), service.config)

    assert service._cache_get("b", StageTimer()) is None
    assert service._cache_get("a", StageTimer()) is not None
    assert service.cache_stats["evictions"] == 1


def test_a_zero_size_cache_stores_nothing() -> None:
    service = _bare_service(cache_size=0)
    service._cache_put("k", _response(), service.config)
    assert service._cache_get("k", StageTimer()) is None
    assert service._cache == {}


def test_clear_cache_drops_everything() -> None:
    service = _bare_service()
    service._cache_put("k", _response(), service.config)
    service.clear_cache()
    assert service._cache_get("k", StageTimer()) is None


def test_the_key_separates_two_different_queries() -> None:
    config = DocuReasonConfig.load_preset("balanced")
    assert QueryService._cache_key("revenue", config) != QueryService._cache_key("margin", config)


def test_the_key_ignores_whitespace_around_the_query() -> None:
    config = DocuReasonConfig.load_preset("balanced")
    assert QueryService._cache_key("  revenue  ", config) == QueryService._cache_key("revenue", config)


def test_an_ablation_switch_changes_the_key() -> None:
    """Otherwise an ablated run could be served the full system's answer."""
    config = DocuReasonConfig.load_preset("balanced")
    ablated = config.apply_query_overrides({"enable_cross_encoder": False})
    assert QueryService._cache_key("revenue", config) != QueryService._cache_key("revenue", ablated)


def test_the_key_ignores_where_artifacts_are_written() -> None:
    """output_dir records where the run put files, not what it answered."""
    config = DocuReasonConfig.load_preset("balanced")
    moved = DocuReasonConfig.load_preset("balanced")
    moved.output_dir = "somewhere/else"
    assert QueryService._cache_key("revenue", config) == QueryService._cache_key("revenue", moved)


# ---------------------------------------------------------------------------
# Per-query overrides must not mutate the service
# ---------------------------------------------------------------------------


def test_overrides_return_a_copy_and_leave_the_original_alone() -> None:
    """One request carrying final_top_k=1 used to narrow the shared service for good."""
    config = DocuReasonConfig.load_preset("balanced")
    original_top_k = config.reranker.final_top_k

    clone = config.apply_query_overrides({"final_top_k": 1, "enable_sparse_retrieval": False})

    assert clone.reranker.final_top_k == 1
    assert clone.retrieval.enable_sparse_retrieval is False
    assert config.reranker.final_top_k == original_top_k
    assert config.retrieval.enable_sparse_retrieval is True


def test_no_overrides_returns_the_same_object() -> None:
    config = DocuReasonConfig.load_preset("balanced")
    assert config.apply_query_overrides({}) is config


def test_an_unknown_override_is_recorded_rather_than_silently_ignored() -> None:
    clone = DocuReasonConfig.load_preset("balanced").apply_query_overrides({"nonsense": 1})
    assert clone.source_map["unknown_overrides"] == "nonsense"


def test_an_uncastable_override_is_rejected_and_recorded() -> None:
    clone = DocuReasonConfig.load_preset("balanced").apply_query_overrides({"final_top_k": "not-a-number"})
    assert "final_top_k" in clone.source_map["unknown_overrides"]
    assert clone.reranker.final_top_k == 5


def test_boolean_overrides_accept_json_and_string_forms() -> None:
    for raw in (False, "false", "0", "no", "off"):
        clone = DocuReasonConfig.load_preset("balanced").apply_query_overrides({"enable_nli": raw})
        assert clone.attribution.enable_nli is False, raw


# ---------------------------------------------------------------------------
# Stage timer
# ---------------------------------------------------------------------------


def test_repeated_stage_names_accumulate() -> None:
    timer = StageTimer()
    for _ in range(3):
        timer.add("dense_search", 5.0)
    payload = timer.as_dict()

    assert payload["stages_ms"]["dense_search"] == 15.0
    assert payload["stage_calls"]["dense_search"] == 3


def test_stages_are_reported_in_pipeline_order() -> None:
    timer = StageTimer()
    for stage in ("generate", "route", "rerank"):
        timer.add(stage, 1.0)
    assert list(timer.as_dict()["stages_ms"]) == ["route", "rerank", "generate"]


def test_an_unaccounted_gap_shows_up_as_a_residual() -> None:
    """A decomposition that does not add up must say so rather than describe 60%."""
    timer = StageTimer()
    timer.add("rerank", 10.0)
    payload = timer.as_dict(total_ms=100.0)

    assert payload["accounted_ms"] == 10.0
    assert payload["residual_ms"] == 90.0
    assert payload["concurrent"] is False


def test_overlapping_stages_report_a_negative_residual() -> None:
    """Under parallel dispatch the stages legitimately exceed wall time."""
    timer = StageTimer()
    timer.add("dense_search", 80.0)
    timer.add("sparse_search", 80.0)
    payload = timer.as_dict(total_ms=100.0)

    assert payload["residual_ms"] < 0
    assert payload["concurrent"] is True


def test_the_context_manager_measures_elapsed_time() -> None:
    timer = StageTimer()
    with timer.stage("rerank"):
        time.sleep(0.02)
    assert timer.as_dict()["stages_ms"]["rerank"] >= 15.0


def test_merging_folds_another_timers_stages_in() -> None:
    first, second = StageTimer(), StageTimer()
    first.add("dense_search", 4.0)
    second.add("dense_search", 6.0)
    second.add("fuse", 1.0)
    first.merge(second)

    assert first.as_dict()["stages_ms"] == {"dense_search": 10.0, "fuse": 1.0}

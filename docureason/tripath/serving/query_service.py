"""query_service.py — the single online query path.

There used to be two implementations of "answer a query": this class (used by every
evaluation script, the notebook, ``BenchmarkRunner`` and the documented public API)
and a separate inline implementation in ``serving/main.py`` (used by the REST
service).  The REST version additionally ran the DuckDB text-to-SQL path, chart
augmentation and parent expansion, so **the evaluated system was not the served
system** — and every eval script read ``response["sql_results"]``, always got
nothing, and silently fell back to fixture presence.

``QueryService`` is now the only implementation.  ``serving/main.py`` delegates to
it, so a change to retrieval, generation or attribution reaches both surfaces at
once.

Response contract
-----------------
``query()`` returns::

    {
      "query": str,
      "route": {...}, "router": {"flags", "probabilities", "weights"},
      "results": [...],              # reranked evidence, final_top_k
      "retrieved_evidence": [...],   # first 6 of results
      "sql_execution": {"executed", "route_active", "reason", "sql_query", "sql_results"},
      "answer": str,                 # the answer text
      "reasoning_chain": str | None,
      "citations": [...],
      "generation_engine": str,
      "generation": {...},           # full generator payload
      "attribution": {...},
      "retrieval_stats": {...},
      "latency_breakdown": {...},    # per-stage ms, plus total and signed residual
      "degradation": [str],          # every fallback that fired, [] on a clean run
      "cached": bool,                # True when served from the response cache
      "preset": str,
    }
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from docureason.tripath.config import DocuReasonConfig
from docureason.tripath.utils import StageTimer, get_logger, trace_execution

from ..attribution.claim_attributor import ClaimSupportAttributor
from ..generation.generate import GenerationModule
from ..ingestion.docling_wrapper import DoclingWrapper
from ..retrieval.hybrid_retriever import HybridRetriever

logger = get_logger(__name__)


class QueryService:
    """An end-to-end multimodal query service for enterprise documents."""

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        config: Optional[DocuReasonConfig | str] = None,
        verbose: bool = True,
    ) -> None:
        validation_info = DocuReasonConfig.validate_inputs(input_dir, output_dir)
        self.input_dir = Path(validation_info["input_dir"])
        self.output_dir = Path(validation_info["output_dir"])

        if isinstance(config, DocuReasonConfig):
            self.config = config
        elif isinstance(config, str):
            self.config = DocuReasonConfig.load_preset(config)
        else:
            self.config = DocuReasonConfig.load_from_yaml("configs/config.yaml")

        # The index builders resolve their directory from config.output_dir, so it
        # must point at this run before any of them is constructed.
        self.config.output_dir = str(self.output_dir)
        self.verbose = verbose
        if self.verbose:
            self.config.print_summary(input_dir=self.input_dir, output_dir=self.output_dir)

        self.documents = DoclingWrapper(input_dir=self.input_dir, output_dir=self.output_dir).ingest()

        # Built once and reused: constructing GenerationModule per request reloaded
        # the local LLM on every call on a GPU machine.
        self.retriever = HybridRetriever(config=self.config)
        self.generator = GenerationModule(config=self.config)
        self.attributor = ClaimSupportAttributor(config=self.config)

        # Response cache. Keyed on the query *and* a fingerprint of the effective
        # config, so a per-query override can never be served a response produced
        # under different parameters. Bounded LRU; the corpus is fixed for the life
        # of this service, and /api/ingest drops the service outright, so a stale
        # corpus cannot be served from here.
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self.cache_stats: Dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def router(self) -> Any:
        """The router owned by the retriever (kept for backward compatibility)."""
        return self.retriever.router

    @property
    def ranker(self) -> Any:
        """The reranker owned by the retriever (kept for backward compatibility)."""
        return self.retriever.ranker

    @trace_execution(logger=logger)
    def query(self, text: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        timer = StageTimer()
        effective_config = self.config.apply_query_overrides(overrides or {})

        cache_key = self._cache_key(text, effective_config)
        cached = self._cache_get(cache_key, timer)
        if cached is not None:
            return cached

        if overrides:
            # A per-query override changes retrieval and generation parameters, so
            # those components are rebuilt; the index handles are reused.
            retriever = HybridRetriever(
                config=effective_config,
                dense=self.retriever.dense,
                sparse=self.retriever.sparse,
            )
            generator = GenerationModule(config=effective_config)
            attributor = ClaimSupportAttributor(config=effective_config)
        else:
            retriever, generator, attributor = self.retriever, self.generator, self.attributor

        router = retriever.router
        with timer.stage("route"):
            route_flags = router.route(text)
            probabilities = router.route_probabilities(text)
            weights = router.get_route_weights(text)

        results = retriever.retrieve(text, self.documents, timer=timer) if self.documents else []
        # The SQL path already ran inside retrieve(), which timed it into the shared
        # timer as the "table_sql" stage; this only shapes the payload.
        sql_payload = self._build_sql_payload(retriever, route_flags, probabilities)

        with timer.stage("generate"):
            generation = generator.generate(text, results, sql_payload)
        answer_text = generation.get("answer", "")
        with timer.stage("attribute"):
            attribution = attributor.attribute(answer_text, results)

        degradation = list(retriever.last_degradation)
        degradation.extend(self._generation_degradation(generation))
        degradation.extend(self._attribution_degradation(attribution))
        if not self.documents:
            degradation.append("corpus:empty")

        response = {
            "query": text,
            "route": route_flags,
            "router": {
                "flags": route_flags,
                "probabilities": probabilities,
                "weights": weights,
            },
            "results": results,
            "retrieved_evidence": results[:6],
            "sql_execution": sql_payload,
            "answer": answer_text,
            "reasoning_chain": generation.get("reasoning_chain"),
            "citations": generation.get("citations", []),
            "generation_engine": generation.get("engine", "unknown"),
            "generation": generation,
            "attribution": attribution,
            "retrieval_stats": dict(retriever.last_stats),
            "latency_breakdown": timer.as_dict(),
            "degradation": degradation,
            "cached": False,
            "preset": effective_config.preset,
        }
        self._cache_put(cache_key, response, effective_config)
        return response

    # ------------------------------------------------------------------
    # Response cache
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(text: str, config: DocuReasonConfig) -> str:
        """Digest of the query and everything about the config that changes the answer.

        ``output_dir`` and ``source_map`` are excluded: they record *where* artifacts
        went and *why* a value was chosen, neither of which affects the response.
        Every other field is included, so an ablation switch or a ``final_top_k``
        override can never be served a response computed under the other setting.
        """
        payload = config.summary_dict()
        payload.pop("source_map", None)
        payload.pop("output_dir", None)
        blob = json.dumps({"query": text.strip(), "config": payload}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_capacity(self, config: Optional[DocuReasonConfig] = None) -> int:
        serving = getattr(config or self.config, "serving", None)
        return max(0, int(getattr(serving, "response_cache_size", 0) or 0))

    def _cache_get(self, key: str, timer: StageTimer) -> Optional[Dict[str, Any]]:
        """A deep copy of the cached response, or None.

        The copy is not defensive pedantry: callers mutate responses (the eval harness
        stamps metrics onto them), and a shared dict would let one query's annotations
        leak into the next caller's answer.

        The returned ``latency_breakdown`` describes *this* call - a cache lookup -
        with the original run's breakdown preserved under
        ``latency_breakdown_first_run``.  Replaying the original timings would quietly
        credit the pipeline with work it did not do and drag every reported percentile
        toward the cache's.
        """
        if self._cache_capacity() == 0:
            return None

        with self._cache_lock:
            stored = self._cache.get(key)
            if stored is None:
                self.cache_stats["misses"] += 1
                return None
            self._cache.move_to_end(key)
            self.cache_stats["hits"] += 1
            response = deepcopy(stored)

        response["cached"] = True
        response["latency_breakdown_first_run"] = stored.get("latency_breakdown")
        response["latency_breakdown"] = timer.as_dict()
        return response

    def _cache_put(self, key: str, response: Dict[str, Any], config: DocuReasonConfig) -> None:
        capacity = self._cache_capacity(config)
        if capacity == 0:
            return
        with self._cache_lock:
            self._cache[key] = deepcopy(response)
            self._cache.move_to_end(key)
            while len(self._cache) > capacity:
                self._cache.popitem(last=False)
                self.cache_stats["evictions"] += 1

    def clear_cache(self) -> None:
        """Drop every cached response.  Call after the corpus or indexes change."""
        with self._cache_lock:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sql_payload(
        retriever: HybridRetriever,
        route_flags: Dict[str, bool],
        probabilities: Dict[str, float],
    ) -> Dict[str, Any]:
        """Build the DuckDB text-to-SQL block from the retriever's own SQL run.

        The SQL path already ran inside ``retriever.retrieve()``; reading its cached
        results here avoids executing every table query twice per request.
        """
        table_active = bool(route_flags.get("table", False))
        sql_results = retriever.last_sql_results if table_active else []
        executed = bool(sql_results)
        return {
            "executed": executed,
            "route_active": table_active,
            "reason": "Table route active"
            if table_active
            else f"Table route inactive (P(table)={probabilities.get('table', 0.0)} below threshold)",
            "sql_query": sql_results[0].get("sql_query", "") if executed else "N/A",
            "sql_results": sql_results[0].get("sql_result", []) if executed else [],
            "mode": sql_results[0].get("mode") if executed else None,
        }

    @staticmethod
    def _generation_degradation(generation: Dict[str, Any]) -> List[str]:
        labels: List[str] = []
        engine = generation.get("engine", "unknown")
        if engine not in ("local_slm_deepseek_r1", "cloud_gemini_api"):
            labels.append(f"generation:{engine}")
        if generation.get("fallback_used"):
            labels.append("generation:fallback_used")
        return labels

    @staticmethod
    def _attribution_degradation(attribution: Dict[str, Any]) -> List[str]:
        """Flag a faithfulness gate that is off, or checking overlap rather than entailment."""
        method = str(attribution.get("method") or "")
        if method == "disabled":
            return ["attribution:disabled"]
        if method and "entailment" not in method:
            return ["attribution:lexical"]
        return []

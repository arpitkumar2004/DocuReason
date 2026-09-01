"""main.py — FastAPI surface over the single :class:`QueryService` query path.

The query endpoint used to contain its own retrieval/generation implementation that
ran the SQL path, chart augmentation and parent expansion, while every evaluation
script used ``QueryService``, which ran none of them.  The two have been unified:
this module now only handles HTTP concerns (validation, payload shape, errors) and
delegates all reasoning to ``QueryService``.

Stateful components are built once in the ``lifespan`` handler instead of in
unguarded module globals mutated from a thread pool, so two concurrent first
requests can no longer both trigger a full ingestion, and the generator is not
reconstructed (and its local model reloaded) per request.

Security note: these routes are **unauthenticated** (audit finding S1, still open).
``input_dir``/``output_dir`` are still accepted from the request body, but are now
contained to the deployment root by :func:`resolve_within_root`, which closes the
traversal and arbitrary-write findings (S2, S3).  Containment is not authorization —
every directory under the root remains reachable by any caller, so do not expose
this service to untrusted networks.  See ``Documentations/audit.md`` §8.
"""

from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from docureason.tripath.utils import get_logger, trace_execution
from docureason.tripath.version import __version__

from ..evaluation.benchmark_dataset import BenchmarkDataset
from ..evaluation.eval_harness import EvaluationHarness
from .query_service import QueryService

logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[3]
UI_DIR = ROOT / "ui"
REPORT_PATH = ROOT / "artifacts" / "test_run" / "pipeline_report.json"

DEFAULT_INPUT_DIR = "samples"
DEFAULT_OUTPUT_DIR = "artifacts/test_run"

_SERVICES: Dict[str, QueryService] = {}
_SERVICES_LOCK = threading.Lock()


class CorpusPathError(ValueError):
    """A request asked for a corpus or artifact directory outside the deployment root."""


def resolve_within_root(candidate: Optional[str], default: str) -> Path:
    """Resolve a request-supplied directory to an absolute path inside :data:`ROOT`.

    ``input_dir`` and ``output_dir`` arrive in the request body, on routes that have
    no authentication.  They were joined onto ``ROOT`` and used as-is, so
    ``"../../.."`` walked out of the deployment and any absolute path ignored
    ``ROOT`` altogether: the query route would ingest any directory the process could
    read, and the ingest route would create directories and write artifacts anywhere
    it could write (audit findings S2 and S3).

    A path must now be relative and must still be inside ``ROOT`` after resolution -
    which also rejects a symlink inside ``ROOT`` that points out of it.  This is
    containment, not authorization: everything under ``ROOT`` remains reachable by
    any caller, so the routes still need a gateway in front of them.
    """
    raw = (candidate or "").strip() or default
    if raw.startswith(("/", "\\")) or Path(raw).is_absolute() or Path(raw).drive:
        raise CorpusPathError(f"{raw!r} must be relative to the deployment root")

    try:
        resolved = (ROOT / raw).resolve()
        root = ROOT.resolve()
    except OSError as exc:  # embedded NULs, over-long names, unreadable mount
        raise CorpusPathError(f"{raw!r} is not a usable path ({exc})") from exc

    if resolved != root and root not in resolved.parents:
        raise CorpusPathError(f"{raw!r} resolves outside the deployment root")
    return resolved


def get_query_service(input_dir: Optional[str] = None, output_dir: Optional[str] = None) -> QueryService:
    """Return the process-wide :class:`QueryService` for a corpus, building it once.

    The lock matters: every endpoint is a synchronous ``def``, so FastAPI runs them
    in a thread pool.  Without it two concurrent first requests each triggered a
    full ingestion.
    """
    in_path = resolve_within_root(input_dir, DEFAULT_INPUT_DIR)
    out_path = resolve_within_root(output_dir, DEFAULT_OUTPUT_DIR)

    key = f"{in_path}::{out_path}"
    service = _SERVICES.get(key)
    if service is not None:
        return service

    with _SERVICES_LOCK:
        service = _SERVICES.get(key)
        if service is None:
            logger.info("Building QueryService for %s", key)
            service = QueryService(input_dir=in_path, output_dir=out_path, verbose=False)
            _SERVICES[key] = service
    return service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm the default corpus at startup; never fail startup because of it."""
    try:
        get_query_service()
        logger.info("Default QueryService warmed at startup")
    except Exception as exc:
        logger.warning("Startup warm-up skipped (%s) — the first request will build it", exc)
    yield
    _SERVICES.clear()


app = FastAPI(title="DocuReason Tri-Path Multimodal RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryPayload(BaseModel):
    query: str
    input_dir: Optional[str] = DEFAULT_INPUT_DIR
    output_dir: Optional[str] = DEFAULT_OUTPUT_DIR
    overrides: Optional[Dict[str, Any]] = None


class IngestPayload(BaseModel):
    input_dir: Optional[str] = DEFAULT_INPUT_DIR
    output_dir: Optional[str] = "artifacts/dashboard_run"


class EvaluatePayload(BaseModel):
    query: str
    relevant_ids: Optional[List[str]] = None


QueryPayload.model_rebuild()
IngestPayload.model_rebuild()
EvaluatePayload.model_rebuild()


@app.get("/health")
@app.get("/api/status")
def status_endpoint() -> Dict[str, str]:
    return {
        "status": "ready",
        "service": "DocuReason Tri-Path Multimodal RAG",
        "version": __version__,
    }


@app.get("/api/report")
def report_endpoint() -> Dict[str, Any]:
    if REPORT_PATH.exists():
        try:
            return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "No pipeline report generated yet. Run ingestion first."}


@app.post("/api/query")
@app.post("/query")
@trace_execution(logger=logger)
def query_endpoint(payload: QueryPayload = Body(...)) -> Dict[str, Any]:
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query string must not be empty")

    try:
        service = get_query_service(payload.input_dir, payload.output_dir)
    except CorpusPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Corpus unavailable: {exc}") from exc

    return service.query(query_text, overrides=payload.overrides)


@app.post("/api/ingest")
@trace_execution(logger=logger)
def ingest_endpoint(payload: IngestPayload) -> Dict[str, Any]:
    from docureason.pipeline import DocuReasonPipeline

    try:
        input_dir = resolve_within_root(payload.input_dir, DEFAULT_INPUT_DIR)
        output_dir = resolve_within_root(payload.output_dir, "artifacts/dashboard_run")
    except CorpusPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        pipeline = DocuReasonPipeline(input_dir=input_dir, output_dir=output_dir)
        report = pipeline.run()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Drop cached services so the next query sees the freshly built corpus/indexes.
    with _SERVICES_LOCK:
        _SERVICES.clear()
    return {"status": "success", "report": report}


@app.post("/evaluate")
def evaluate_endpoint(payload: EvaluatePayload) -> Dict[str, Any]:
    service = get_query_service(DEFAULT_INPUT_DIR, "artifacts/serving")
    harness = EvaluationHarness(output_dir=ROOT / "artifacts/serving")
    response = service.query(payload.query)
    metrics = harness.evaluate(payload.query, response["results"], relevant_ids=payload.relevant_ids or [])
    return {"query": payload.query, "metrics": metrics, "results": response["results"]}


@app.get("/benchmarks")
def benchmarks_endpoint() -> List[Dict[str, Any]]:
    return BenchmarkDataset().build()


# Mount static single-page UI directory if present
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

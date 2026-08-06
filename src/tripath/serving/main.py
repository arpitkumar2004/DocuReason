from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.tripath.utils import get_logger, trace_execution
from src.tripath.router.configurable_router import ConfigurableRouter
from src.tripath.retrieval.hybrid_retriever import HybridRetriever
from src.tripath.retrieval.table_sql import TableSQLRetriever
from src.tripath.generation.generate import GenerationModule
from src.tripath.attribution.nli_attributor import NLIFaithfulnessAttributor
from src.tripath.ingestion.docling_wrapper import DoclingWrapper
from docureason.pipeline import DocuReasonPipeline
from .async_query_service import AsyncQueryService
from .query_service import QueryService
from ..evaluation.benchmark_dataset import BenchmarkDataset
from ..evaluation.eval_harness import EvaluationHarness

logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[3]
UI_DIR = ROOT / "ui"
REPORT_PATH = ROOT / "artifacts" / "test_run" / "pipeline_report.json"

app = FastAPI(title="DocuReason Tri-Path Multimodal RAG API")

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Cached Documents & Retrievers
_CACHED_DOCUMENTS: Optional[List[Any]] = None
_HYBRID_RETRIEVER: Optional[HybridRetriever] = None
_TABLE_SQL_RETRIEVER: Optional[TableSQLRetriever] = None


class QueryPayload(BaseModel):
    query: str
    input_dir: Optional[str] = "samples"
    output_dir: Optional[str] = "artifacts/test_run"


class IngestPayload(BaseModel):
    input_dir: Optional[str] = "samples"
    output_dir: Optional[str] = "artifacts/dashboard_run"


class EvaluatePayload(BaseModel):
    query: str
    relevant_ids: Optional[List[str]] = None


@app.get("/health")
@app.get("/api/status")
def status_endpoint() -> Dict[str, str]:
    return {
        "status": "ready",
        "service": "DocuReason Tri-Path Multimodal RAG",
        "version": "1.1.0",
    }


@app.get("/api/report")
def report_endpoint() -> Dict[str, Any]:
    if REPORT_PATH.exists():
        try:
            return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "No pipeline report generated yet. Run ingestion first."}


@app.post("/api/query")
@app.post("/query")
@trace_execution(logger=logger)
def query_endpoint(payload: QueryPayload) -> Dict[str, Any]:
    global _CACHED_DOCUMENTS
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query string must not be empty")

    # 1. Soft Multi-Label Router Probabilities & Weights
    router = ConfigurableRouter()
    route_flags = router.route(query_text)
    probs = router.route_probabilities(query_text)
    weights = router.get_route_weights(query_text)

    # 2. In-Memory Cached Document Retrieval
    if _CACHED_DOCUMENTS is None:
        docs_dir = ROOT / (payload.input_dir or "samples")
        if docs_dir.exists():
            try:
                _CACHED_DOCUMENTS = DoclingWrapper(input_dir=docs_dir, output_dir=ROOT / "artifacts" / "test_run").ingest(force_reingest=False)
            except Exception as exc:
                logger.warning("Document ingestion fallback: %s", exc)
                _CACHED_DOCUMENTS = []

    global _HYBRID_RETRIEVER, _TABLE_SQL_RETRIEVER
    if _HYBRID_RETRIEVER is None:
        _HYBRID_RETRIEVER = HybridRetriever()
    if _TABLE_SQL_RETRIEVER is None:
        _TABLE_SQL_RETRIEVER = TableSQLRetriever()

    documents = _CACHED_DOCUMENTS or []
    retrieved_candidates = _HYBRID_RETRIEVER.retrieve(query_text, documents) if documents else []

    # 3. DuckDB Text-to-SQL Execution (Only executed if table route threshold is met)
    is_table_active = route_flags.get("table", False)
    if is_table_active and documents:
        sql_results = _TABLE_SQL_RETRIEVER.retrieve(query_text, documents)
    else:
        sql_results = []

    # 4. Multi-Modal SLM/LLM Generation & NLI Attribution Verification
    generator = GenerationModule()
    sql_payload = {
        "executed": is_table_active and len(sql_results) > 0,
        "sql_query": sql_results[0].get("sql_query", "") if (is_table_active and sql_results) else "N/A",
        "sql_results": sql_results[0].get("sql_result", []) if (is_table_active and sql_results) else [],
    }
    gen_result = generator.generate(query_text, retrieved_candidates, sql_payload)
    answer_text = gen_result.get("answer", "")
    reasoning_chain = gen_result.get("reasoning_chain")
    citations = gen_result.get("citations", [])

    attributor = NLIFaithfulnessAttributor()
    attribution = attributor.attribute(answer_text, retrieved_candidates)

    return {
        "query": query_text,
        "router": {
            "flags": route_flags,
            "probabilities": probs,
            "weights": weights,
        },
        "sql_execution": {
            "executed": sql_payload["executed"],
            "route_active": is_table_active,
            "reason": "Table route active" if is_table_active else f"Table route inactive (P(table)={probs.get('table', 0.0)} < threshold 0.35)",
            "sql_query": sql_payload["sql_query"],
            "sql_results": sql_payload["sql_results"],
        },
        "results": retrieved_candidates,
        "retrieved_evidence": retrieved_candidates[:6],
        "answer": answer_text,
        "reasoning_chain": reasoning_chain,
        "citations": citations,
        "generation_engine": gen_result.get("engine", "unknown"),
        "attribution": attribution,
    }


@app.post("/api/ingest")
@trace_execution(logger=logger)
def ingest_endpoint(payload: IngestPayload) -> Dict[str, Any]:
    global _CACHED_DOCUMENTS
    input_dir = payload.input_dir or "samples"
    output_dir = payload.output_dir or "artifacts/dashboard_run"
    try:
        pipeline = DocuReasonPipeline(input_dir=input_dir, output_dir=output_dir)
        report = pipeline.run()
        _CACHED_DOCUMENTS = None  # Reset document cache after new ingestion
        return {"status": "success", "report": report}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate")
def evaluate_endpoint(payload: EvaluatePayload) -> Dict[str, Any]:
    query_service = QueryService(input_dir=ROOT / "samples", output_dir=ROOT / "artifacts/serving")
    harness = EvaluationHarness(output_dir=ROOT / "artifacts/serving")
    response = query_service.query(payload.query)
    metrics = harness.evaluate(payload.query, response["results"], relevant_ids=payload.relevant_ids or [])
    return {"query": payload.query, "metrics": metrics, "results": response["results"]}


@app.get("/benchmarks")
def benchmarks_endpoint() -> List[Dict[str, Any]]:
    return BenchmarkDataset().build()


# Mount static single-page UI directory if present
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

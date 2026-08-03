from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..evaluation.benchmark_dataset import BenchmarkDataset
from ..evaluation.eval_harness import EvaluationHarness
from .async_query_service import AsyncQueryService
from .query_service import QueryService

app = FastAPI(title="DocuReason Tri-Path")


class QueryRequest(BaseModel):
    query: str


class EvaluateRequest(BaseModel):
    query: str
    relevant_ids: list[str] | None = None


service = QueryService(input_dir=Path("samples"), output_dir=Path("artifacts/phase2"))
async_service = AsyncQueryService(input_dir=Path("samples"), output_dir=Path("artifacts/phase2"))
harness = EvaluationHarness(output_dir=Path("artifacts/phase2"))
benchmark_dataset = BenchmarkDataset()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    template_path = Path("templates") / "index.html"
    return template_path.read_text(encoding="utf-8")


@app.post("/query")
def query(request: QueryRequest) -> dict:
    return async_service.run(request.query)


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict:
    response = service.query(request.query)
    metrics = harness.evaluate(request.query, response["results"], relevant_ids=request.relevant_ids or [])
    return {"query": request.query, "metrics": metrics, "results": response["results"]}


@app.get("/benchmarks")
def benchmarks() -> list[dict]:
    return benchmark_dataset.build()

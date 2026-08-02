from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from .query_service import QueryService

app = FastAPI(title="DocuReason Tri-Path")


class QueryRequest(BaseModel):
    query: str


service = QueryService(input_dir=Path("samples"), output_dir=Path("artifacts/phase2"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    return service.query(request.query)

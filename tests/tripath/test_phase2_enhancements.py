from pathlib import Path

import pytest

from src.tripath.fusion.late_fusion import ReciprocalRankFuser
from src.tripath.ingestion.schema import Document, Region
from src.tripath.retrieval.hybrid_retriever import HybridRetriever
from src.tripath.retrieval.ranker import Ranker
from src.tripath.retrieval.table_sql import TableSQLRetriever
from src.tripath.router.configurable_router import ConfigurableRouter
from src.tripath.router.infer_router import Router
from src.tripath.router.train_router import RouterTrainer


def test_configurable_router_probabilities_and_weights():
    router = ConfigurableRouter()
    query = "revenue by region and quarterly table"
    probs = router.route_probabilities(query)
    weights = router.get_route_weights(query)

    assert "text" in probs and "table" in probs and "vision" in probs
    assert probs["table"] > 0.5
    assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)


def test_router_trainer_and_export(tmp_path: Path):
    model_file = tmp_path / "router_model.json"
    trainer = RouterTrainer()
    output_path = trainer.train_and_export(model_file)

    assert output_path.exists()

    infer_router = Router(model_path=output_path)
    probs = infer_router.route_probabilities("revenue report")
    assert probs["text"] > 0.0


def test_table_sql_duckdb_and_fallback():
    retriever = TableSQLRetriever()
    doc = Document(
        id="doc-sql-test",
        source="sample.pdf",
        title="Financial Test",
        regions=[
            Region(
                type="table",
                text="Quarter | Region | Revenue\nQ1 | North America | 10000\nQ2 | Europe | 8000",
                start=0,
                end=100,
                table_markdown="Quarter | Region | Revenue\nQ1 | North America | 10000\nQ2 | Europe | 8000",
                table_json={"columns": ["Quarter", "Region", "Revenue"], "rows": [["Q1", "North America", "10000"], ["Q2", "Europe", "8000"]]},
            )
        ],
    )

    results = retriever.retrieve("revenue by region total sum", [doc])
    assert len(results) >= 1
    assert results[0]["score"] > 0.0
    assert "sql_query" in results[0]
    assert results[0]["modality"] == "table"


def test_reciprocal_rank_fusion_rrf():
    fuser = ReciprocalRankFuser(rrf_k=60)
    runs = {
        "text": [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.7}],
        "table": [{"id": "doc2", "score": 0.95}, {"id": "doc3", "score": 0.6}],
    }
    fused = fuser.fuse_rrf(runs)

    assert len(fused) == 3
    # doc2 appeared in both runs, should have higher RRF score
    assert fused[0]["id"] == "doc2"
    assert fused[0]["score"] > 0.0


def test_cross_encoder_ranker():
    ranker = Ranker()
    candidates = [
        {"id": "c1", "text": "Unrelated generic text about weather", "score": 0.5},
        {"id": "c2", "text": "Quarterly revenue report breakdown by region", "score": 0.8},
    ]

    reranked = ranker.rank("quarterly revenue report", candidates)
    assert len(reranked) == 2
    assert reranked[0]["id"] == "c2"


def test_hybrid_retriever_end_to_end_flow():
    retriever = HybridRetriever()
    doc = Document(
        id="doc-retrieval-test",
        source="test.pdf",
        title="Quarterly Report",
        regions=[
            Region(type="title", text="Quarterly Revenue Report", start=0, end=24),
            Region(
                type="table",
                text="Quarter | Region | Revenue\nQ1 | West | 15000",
                start=25,
                end=80,
                table_json={"columns": ["Quarter", "Region", "Revenue"], "rows": [["Q1", "West", "15000"]]},
            ),
        ],
    )

    results = retriever.retrieve("revenue by region", [doc])
    assert len(results) >= 1
    assert results[0]["document_id"] == "doc-retrieval-test"
    assert "rank_score" in results[0]

from pathlib import Path

from src.tripath.evaluation.eval_harness import EvaluationHarness
from src.tripath.ingestion.docling_wrapper import DoclingWrapper
from src.tripath.retrieval.hybrid_retriever import HybridRetriever
from src.tripath.router.train_router import RouterTrainer


def test_retrieval_research_stack_components(tmp_path):
    sample_dir = Path(__file__).resolve().parents[2] / "samples"
    output_dir = tmp_path / "research-output"

    documents = DoclingWrapper(input_dir=sample_dir, output_dir=output_dir).ingest()
    results = HybridRetriever().retrieve("revenue by region", documents)

    assert results
    assert any(result.get("modality") == "table" for result in results)

    metrics = EvaluationHarness(output_dir=tmp_path).evaluate(
        "revenue by region",
        results,
        relevant_ids=[documents[0].id],
    )
    assert metrics["result_count"] >= 1
    assert "ndcg_at_5" in metrics

    dataset_path = RouterTrainer().save_dataset(tmp_path / "router_dataset.json")
    assert dataset_path.exists()

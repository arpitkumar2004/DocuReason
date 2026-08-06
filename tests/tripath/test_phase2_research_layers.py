from pathlib import Path

from src.tripath.evaluation.eval_harness import EvaluationHarness
from src.tripath.retrieval.hybrid_retriever import HybridRetriever
from src.tripath.router.train_router import RouterTrainer
from src.tripath.ingestion.docling_wrapper import DoclingWrapper


def test_research_ready_retrieval_components(tmp_path):
    sample_dir = Path(__file__).resolve().parents[2] / "samples"
    output_dir = tmp_path / "research-output"

    documents = DoclingWrapper(input_dir=sample_dir, output_dir=output_dir).ingest()
    results = HybridRetriever().retrieve("revenue by region", documents)
    assert results

    harness = EvaluationHarness(output_dir=tmp_path)
    metrics = harness.evaluate("revenue by region", results, relevant_ids=[documents[0].id])
    assert metrics["result_count"] >= 1

    dataset_path = RouterTrainer().save_dataset(tmp_path / "router_dataset.json")
    assert dataset_path.exists()

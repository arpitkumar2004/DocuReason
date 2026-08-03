from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT, ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tripath.ingestion.docling_wrapper import DoclingWrapper
from tripath.ingestion.chunker import SectionAwareChunker
from tripath.retrieval.hybrid_retriever import HybridRetriever
from tripath.evaluation.eval_harness import EvaluationHarness
from tripath.evaluation.benchmark_dataset import BenchmarkDataset
from tripath.router.configurable_router import ConfigurableRouter


def main() -> None:
    sample_dir = ROOT / "samples"
    output_dir = ROOT / "artifacts" / "test_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = DoclingWrapper(input_dir=sample_dir, output_dir=output_dir).ingest()
    chunker = SectionAwareChunker(chunk_size=40, overlap=8)
    all_chunks = []
    for document in documents:
        all_chunks.extend(chunker.chunk_document(document))

    retriever = HybridRetriever()
    results = retriever.retrieve("revenue by region", documents)
    router = ConfigurableRouter()
    route = router.route("revenue by region")
    harness = EvaluationHarness(output_dir=output_dir)
    metrics = harness.evaluate("revenue by region", results, relevant_ids=[documents[0].id])

    payload = {
        "document_count": len(documents),
        "chunk_count": len(all_chunks),
        "route": route,
        "retrieval_results": results[:5],
        "metrics": metrics,
        "benchmark_dataset": BenchmarkDataset().build(),
    }

    (output_dir / "pipeline_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT, ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tripath.utils import setup_logger, trace_execution, trace_pipeline_stage
from tripath.ingestion.docling_wrapper import DoclingWrapper
from tripath.ingestion.chunker import SectionAwareChunker
from tripath.retrieval.hybrid_retriever import HybridRetriever
from tripath.evaluation.eval_harness import EvaluationHarness
from tripath.evaluation.benchmark_dataset import BenchmarkDataset
from tripath.router.configurable_router import ConfigurableRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full end-to-end TriPath research pipeline.")
    parser.add_argument("--input-dir", type=str, default=str(ROOT / "samples"), help="Directory containing sample files.")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "test_run"), help="Output directory for pipeline artifacts.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console log verbosity level.")
    parser.add_argument("--log-file", type=str, default=None, help="File path to write detailed execution logs.")
    parser.add_argument("--force-reingest", action="store_true", help="Force re-running document parsing and index building.")
    return parser.parse_args()


@trace_pipeline_stage("End-to-End TriPath Pipeline Execution")
def main() -> None:
    args = parse_args()
    logger = setup_logger("scripts.run_pipeline", log_level=args.log_level, log_file=args.log_file)
    
    sample_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Pipeline starting with input_dir=%s, output_dir=%s", sample_dir, output_dir)

    documents = DoclingWrapper(input_dir=sample_dir, output_dir=output_dir).ingest(force_reingest=args.force_reingest)
    logger.info("Ingested %d document(s)", len(documents))

    chunker = SectionAwareChunker(chunk_size=40, overlap=8)
    all_chunks = []
    for document in documents:
        chunks = chunker.chunk_document(document)
        all_chunks.extend(chunks)
        logger.debug("Document %s chunked into %d chunks", document.id, len(chunks))

    retriever = HybridRetriever()
    query = "revenue by region"
    results = retriever.retrieve(query, documents)
    logger.info("Retrieved %d candidate document(s) for query '%s'", len(results), query)

    router = ConfigurableRouter()
    route = router.route(query)
    logger.info("Router decision for '%s': %s", query, route)

    harness = EvaluationHarness(output_dir=output_dir)
    relevant_ids = [documents[0].id] if documents else []
    metrics = harness.evaluate(query, results, relevant_ids=relevant_ids)
    logger.info("Evaluation metrics computed: %s", metrics)

    payload = {
        "document_count": len(documents),
        "chunk_count": len(all_chunks),
        "route": route,
        "retrieval_results": results[:5],
        "metrics": metrics,
        "benchmark_dataset": BenchmarkDataset().build(),
    }

    report_path = output_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Pipeline report written to %s", report_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from src.tripath.utils import get_logger, setup_logger
from .pipeline import DocuReasonPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the document ingestion and indexing pipeline")
    parser.add_argument("--input-dir", default="samples", help="Input directory containing documents")
    parser.add_argument("--output-dir", default="artifacts/index", help="Output directory for generated corpus and index")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console log verbosity level")
    parser.add_argument("--log-file", default=None, help="File path to write detailed execution logs")
    args = parser.parse_args()

    setup_logger("docureason", log_level=args.log_level, log_file=args.log_file)
    logger = get_logger("docureason.cli")
    logger.info("DocuReason CLI triggered with input_dir=%s, output_dir=%s", args.input_dir, args.output_dir)

    pipeline = DocuReasonPipeline(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    result = pipeline.run()
    logger.info("Pipeline run complete: %s documents, %s chunks", result['document_count'], result['chunk_count'])
    print(f"Processed {result['document_count']} documents into {result['chunk_count']} chunks")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import Phase1Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1 ingestion pipeline")
    parser.add_argument("--input-dir", default="samples")
    parser.add_argument("--output-dir", default="artifacts/phase1")
    args = parser.parse_args()

    pipeline = Phase1Pipeline(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))
    result = pipeline.run()
    print(f"Processed {result['document_count']} documents into {result['chunk_count']} chunks")


if __name__ == "__main__":
    main()

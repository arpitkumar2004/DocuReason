"""DocuReason Quickstart: Ingestion and Indexing Pipeline."""

from docureason import DocuReasonPipeline


def main() -> None:
    # Initialize offline ingestion pipeline
    pipeline = DocuReasonPipeline(
        input_dir="samples",
        output_dir="artifacts/quickstart_run"
    )

    # Run layout segmentation, table processing, and vector indexing
    report = pipeline.run()

    print("Pipeline Execution Summary:")
    print(f"Status: {report['status']}")
    print(f"Processed Documents: {report['document_count']}")
    print(f"Generated Chunks: {report['chunk_count']}")
    print(f"Dense Vector Count: {report['dense_vectors']}")
    print(f"Sparse Terms Count: {report['sparse_terms']}")


if __name__ == "__main__":
    main()

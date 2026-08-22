"""DocuReason Quickstart: Ingestion and Indexing Pipeline with Configuration Transparency."""

from docureason import DocuReasonConfig, DocuReasonPipeline, MissingRequiredConfigError, InvalidCorpusError


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Initialize pipeline with minimum required developer inputs:
    #    - input_dir  : Path to local document corpus directory (Required)
    #    - output_dir : Path to target artifact output directory (Required)
    #    - config     : Optional profile preset ('quality_max', 'balanced', etc.)
    # -------------------------------------------------------------------------
    try:
        pipeline = DocuReasonPipeline(
            input_dir="samples",
            output_dir="artifacts/quickstart_run",
            config="balanced",
            verbose=True,  # Automatically logs the Configuration Transparency Table
        )

        # Access loaded framework configuration details
        print(f"\n[Active Profile Preset]: {pipeline.config.preset}")
        print(f"[Dense Vector Model]   : {pipeline.config.indexing.model_name or 'Default MiniLM'}")

        # Execute layout segmentation, table processing, and vector indexing
        report = pipeline.run()

        print("\nPipeline Execution Summary:")
        print(f"Processed Documents: {report.get('document_count', 0)}")
        print(f"Generated Chunks   : {report.get('chunk_count', 0)}")
        print(f"Artifact Target    : {report.get('output_dir')}")

    except (MissingRequiredConfigError, InvalidCorpusError) as exc:
        print(f"\n[DocuReason Input Validation Failed]:\n{exc}")


if __name__ == "__main__":
    main()


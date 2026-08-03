from pathlib import Path

from docureason.pipeline import Phase1Pipeline
from src.tripath.ingestion.format_loader import FormatAwareLoader


def test_format_aware_loader_supports_common_documents(tmp_path):
    sample_dir = Path(__file__).resolve().parents[2] / "samples"
    output_dir = tmp_path / "format-output"

    pipeline = Phase1Pipeline(input_dir=sample_dir, output_dir=output_dir)
    result = pipeline.run()

    assert result["document_count"] >= 1

    payload = FormatAwareLoader().load(sample_dir / "sample_doc_1.txt")
    assert payload["supported"] is True
    assert payload["text"]

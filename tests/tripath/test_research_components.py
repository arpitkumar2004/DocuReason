from pathlib import Path

from src.tripath.evaluation.benchmark_dataset import BenchmarkDataset
from src.tripath.ingestion.chunker import SectionAwareChunker
from src.tripath.ingestion.docling_wrapper import DoclingWrapper
from src.tripath.ingestion.schema import Document


def test_chunking_and_benchmark_dataset(tmp_path):
    sample_dir = Path(__file__).resolve().parents[2] / "samples"
    output_dir = tmp_path / "research-output"

    documents = DoclingWrapper(input_dir=sample_dir, output_dir=output_dir).ingest()
    chunker = SectionAwareChunker(chunk_size=20, overlap=5)
    chunks = chunker.chunk_document(documents[0])

    assert chunks
    assert any(chunk.metadata and chunk.metadata.get("section") for chunk in chunks)

    dataset_path = BenchmarkDataset().save(tmp_path / "benchmark_dataset.json")
    assert dataset_path.exists()

from pathlib import Path

from src.tripath.ingestion.docling_wrapper import DoclingWrapper
from src.tripath.ingestion.modality_splitter import ModalitySplitter
from src.tripath.indexing.text_encoder import TextEncoder


def test_ingestion_and_indexing_modules(tmp_path):
    sample_dir = Path(__file__).resolve().parents[2] / "samples"
    output_dir = tmp_path / "tripath-output"

    wrapper = DoclingWrapper(input_dir=sample_dir, output_dir=output_dir)
    documents = wrapper.ingest()

    assert len(documents) == 2
    assert documents[0].title == "Quarterly Revenue Report"
    assert any(region.type == "table" for region in documents[0].regions)
    assert any(region.type == "figure" for region in documents[1].regions)

    splitter = ModalitySplitter()
    split_regions = splitter.split(documents[0])
    assert set(split_regions.keys()) == {"text", "table", "vision"}
    assert split_regions["table"]

    payload = TextEncoder().encode(documents)
    assert payload

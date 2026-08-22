from pathlib import Path

from src.tripath.indexing.text_encoder import TextEncoder
from src.tripath.ingestion.docling_wrapper import DoclingWrapper
from src.tripath.ingestion.modality_splitter import ModalitySplitter


def test_ingestion_and_indexing_modules(tmp_path):
    sample_dir = Path(__file__).resolve().parents[2] / "samples"
    output_dir = tmp_path / "tripath-output"

    wrapper = DoclingWrapper(input_dir=sample_dir, output_dir=output_dir)
    documents = wrapper.ingest()

    assert len(documents) >= 2
    doc_map = {d.source: d for d in documents}
    target_doc = doc_map.get("sample_doc_1.txt", documents[0])
    assert target_doc.title == "Quarterly Revenue Report"
    assert any(region.type == "table" for region in target_doc.regions)

    splitter = ModalitySplitter()
    split_regions = splitter.split(target_doc)
    assert set(split_regions.keys()) == {"text", "table", "vision"}
    assert split_regions["table"]

    payload = TextEncoder().encode(documents)
    assert payload

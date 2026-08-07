from pathlib import Path

from src.tripath.ingestion.docling_wrapper import DoclingWrapper
from src.tripath.retrieval.table_retrieval import TableRetrieval
from src.tripath.retrieval.text_retrieval import TextRetrieval
from src.tripath.retrieval.vision_retrieval import VisionRetrieval
from src.tripath.router.infer_router import Router


def test_multimodal_retrieval_and_routing(tmp_path):
    sample_dir = Path(__file__).resolve().parents[2] / "samples"
    output_dir = tmp_path / "tripath-output"

    documents = DoclingWrapper(input_dir=sample_dir, output_dir=output_dir).ingest()

    text_results = TextRetrieval().retrieve("revenue growth", documents)
    table_results = TableRetrieval().retrieve("revenue by region", documents)
    vision_results = VisionRetrieval().retrieve("adoption chart", documents)

    assert text_results
    assert table_results
    assert vision_results

    router = Router()
    route = router.route("revenue by region")
    assert route["text"] is True
    assert route["table"] is True

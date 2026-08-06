from pathlib import Path

from src.tripath.serving.async_query_service import AsyncQueryService


def test_async_query_service_components(tmp_path):
    service = AsyncQueryService(input_dir=Path("samples"), output_dir=tmp_path / "serving-output")
    response = service.run("revenue by region")

    assert response["answer"]
    assert response["citation_report"]["status"] in {"verified", "needs_review"}
    assert response["citation_report"]["attribution_precision"] >= 0.0
    assert response["fused_results"]
    assert response["metrics"]["result_count"] >= 1

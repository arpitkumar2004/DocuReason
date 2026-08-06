from pathlib import Path

from src.tripath.serving.query_service import QueryService


def test_query_service_returns_ranked_results(tmp_path):
    service = QueryService(input_dir=Path("samples"), output_dir=tmp_path / "service-output")
    response = service.query("revenue by region")

    assert response["query"] == "revenue by region"
    assert response["route"]["text"] is True
    assert response["route"]["table"] is True
    assert response["results"]
    assert response["embeddings"]

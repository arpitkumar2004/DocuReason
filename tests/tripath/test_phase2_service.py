from pathlib import Path

from docureason.tripath.serving.query_service import QueryService


def test_query_service_returns_ranked_results(tmp_path):
    service = QueryService(input_dir=Path("samples"), output_dir=tmp_path / "service-output")
    response = service.query("revenue by region")

    assert response["query"] == "revenue by region"
    assert response["route"]["text"] is True
    assert response["route"]["table"] is True
    assert response["results"]
    # "embeddings" was removed in v1.1.4: it held non-semantic hash vectors
    # recomputed over the whole corpus on every query and never consumed.
    assert "embeddings" not in response

from fastapi.testclient import TestClient

from src.tripath.serving.main import app


def test_query_and_evaluation_endpoints():
    client = TestClient(app)

    query_response = client.post("/query", json={"query": "revenue by region"})
    assert query_response.status_code == 200
    query_payload = query_response.json()
    assert query_payload["answer"]
    assert query_payload["results"]

    evaluate_response = client.post(
        "/evaluate",
        json={"query": "revenue by region", "relevant_ids": [query_payload["results"][0]["document_id"]]},
    )
    assert evaluate_response.status_code == 200
    evaluate_payload = evaluate_response.json()
    assert evaluate_payload["metrics"]["result_count"] >= 1

    benchmarks_response = client.get("/benchmarks")
    assert benchmarks_response.status_code == 200
    assert benchmarks_response.json()

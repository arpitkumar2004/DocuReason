from fastapi.testclient import TestClient

import src.tripath.serving.main as main_module
from src.tripath.ingestion.schema import Document, Region


def test_query_and_evaluation_endpoints():
    # Pre-seed cached documents to eliminate PDF layout parsing overhead in unit tests
    main_module._CACHED_DOCUMENTS = [
        Document(
            id="sample_doc_1",
            title="Revenue Report",
            source="sample_doc_1.txt",
            regions=[
                Region(
                    type="paragraph",
                    text="[Context: Revenue Report] Q1 revenue by region totaled $10 million in North America.",
                    bbox=(0, 0, 100, 50),
                )
            ],
        )
    ]

    client = TestClient(main_module.app)

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

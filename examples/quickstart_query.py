"""DocuReason Quickstart: Online Query Serving & Multimodal Retrieval."""

from docureason.serving import QueryService


def main() -> None:
    # Initialize query serving engine
    service = QueryService(
        input_dir="samples",
        output_dir="artifacts/quickstart_run"
    )

    # Execute query
    query = "What was the revenue growth shown in the financial comparison table?"
    response = service.query(query)

    print(f"Query: {query}")
    print(f"Answer: {response['answer']}")
    print(f"Routing Decision: {response['route']}")
    print(f"Retrieved Candidates: {len(response['results'])}")
    if response["results"]:
        top_hit = response["results"][0]
        print(f"Top Document ID: {top_hit['document_id']}")


if __name__ == "__main__":
    main()

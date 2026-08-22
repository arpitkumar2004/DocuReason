"""DocuReason Quickstart: Online Query Serving & Multimodal Retrieval with Configuration Transparency."""

from docureason import DocuReasonConfig, QueryService


def main() -> None:
    # Initialize query serving engine with minimum inputs and transparency enabled
    service = QueryService(
        input_dir="samples",
        output_dir="artifacts/quickstart_run",
        config="balanced",
        verbose=True,
    )

    # Print explicit configuration summary on demand
    service.config.print_summary(input_dir=service.input_dir, output_dir=service.output_dir)

    query = "What was the revenue growth shown in the financial comparison table?"
    response = service.query(query)

    print(f"\nQuery              : {query}")
    print(f"Answer             : {response.get('answer')}")
    print(f"Routing Decision   : {response.get('route')}")
    print(f"Retrieved Candidates: {len(response.get('results', []))}")


if __name__ == "__main__":
    main()


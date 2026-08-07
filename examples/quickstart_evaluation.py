"""DocuReason Quickstart: Evaluation & NLI Attribution Precision."""

from src.tripath.evaluation import DatasetExporter, EvaluationHarness


def main() -> None:
    # Initialize evaluation harness
    harness = EvaluationHarness(output_dir="artifacts/quickstart_run")

    sample_query = "What is the operating margin?"
    sample_results = [
        {"document_id": "sample_doc_1", "text": "Operating margin reached 21% in Q3."}
    ]

    # Evaluate single query
    metrics = harness.evaluate_single(
        query=sample_query,
        results=sample_results,
        relevant_ids=["sample_doc_1"],
        answer="Operating margin was 21%.",
        evidence=sample_results
    )

    print("Evaluation Metrics:")
    print(f"Recall@5: {metrics.get('recall_at_5')}")
    print(f"nDCG@5: {metrics.get('ndcg_at_5')}")
    print(f"Attribution Precision: {metrics.get('attribution_precision')}")
    print(f"Attribution Status: {metrics.get('status')}")

    # Export fine-tuning dataset
    exporter = DatasetExporter(output_dir="artifacts/quickstart_run")
    export_path = exporter.export_fine_tuning_dataset(output_format="jsonl")
    print(f"Fine-tuning dataset exported to: {export_path}")


if __name__ == "__main__":
    main()

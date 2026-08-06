import json
from pathlib import Path

from docureason.pipeline import DocuReasonPipeline
from src.tripath.evaluation.mlflow_tracker import MLflowTracker
from src.tripath.serving.query_service import QueryService


def test_pipeline_artifacts_and_query_answer(tmp_path):
    sample_dir = Path(__file__).resolve().parents[2] / "samples"
    output_dir = tmp_path / "artifact-run"

    result = DocuReasonPipeline(input_dir=sample_dir, output_dir=output_dir).run()

    assert result["document_count"] >= 2
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["documents"]
    assert manifest["chunks"]
    assert manifest["documents"][0]["metadata"]["namespace"] == "tripath-v1"

    service = QueryService(input_dir=sample_dir, output_dir=tmp_path / "service-output")
    response = service.query("revenue by region")
    assert "answer" in response
    assert response["answer"]

    tracker = MLflowTracker(output_dir=tmp_path / "runs")
    metrics_path = tracker.log_metrics({"recall_at_5": 1.0}, run_name="test-run")
    assert metrics_path.exists()

from pathlib import Path

from src.tripath.evaluation.ablation import AblationStudy
from src.tripath.evaluation.benchmark_runner import BenchmarkRunner


def test_phase4_benchmark_and_ablation(tmp_path):
    runner = BenchmarkRunner(input_dir=Path("samples"), output_dir=tmp_path / "phase4-output")
    payload = runner.run_suite(["revenue by region", "adoption chart"])

    assert payload["query_count"] == 2
    assert payload["summary"]["average_recall_at_5"] >= 0.0

    baseline = payload
    ablation_payload = AblationStudy().run(baseline, [{"name": "no-fusion", "summary": {"average_recall_at_5": 0.1}}])
    assert ablation_payload["ablations"][0]["name"] == "no-fusion"
    assert ablation_payload["ablations"][0]["delta_vs_full"] >= 0.0

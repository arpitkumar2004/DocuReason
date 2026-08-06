import pytest
from src.tripath.retrieval.chart_understanding import ChartUnderstandingModule
from src.tripath.generation.generate import GenerationModule


def test_dynamic_chart_linearization():
    chart_module = ChartUnderstandingModule()
    evidence = [
        {
            "id": "doc1-fig1",
            "modality": "vision",
            "text": "Quarterly adoption trend graph showing growth across Q1 to Q4",
            "metadata": {"page_no": 4, "clip_chart_type": "bar_chart"},
        },
        {
            "id": "doc1-text1",
            "modality": "text",
            "text": "Regular text passage about revenue.",
        },
    ]

    understood = chart_module.understand("adoption chart trend", evidence)
    assert len(understood) == 1
    assert understood[0]["chart_type"] == "bar_chart"
    assert "[BAR_CHART - Page 4]" in understood[0]["linearized"]
    assert "Quarterly adoption" in understood[0]["linearized"]


def test_deepseek_r1_reasoning_extraction():
    raw_response = (
        "<think>\n"
        "Analyzing balance sheet figures for 2019:\n"
        "1. Cash & equivalents = 1,250\n"
        "2. Total Assets = 5,000\n"
        "</think>\n\n"
        "Based on the 2019 balance sheet, total assets were **5,000** [Source: sample_doc_1]."
    )
    reasoning, clean_answer = GenerationModule._extract_reasoning_chain(raw_response)
    assert reasoning is not None
    assert "Analyzing balance sheet figures for 2019" in reasoning
    assert "<think>" not in clean_answer
    assert "total assets were **5,000**" in clean_answer


def test_generation_module_offline_template():
    gen = GenerationModule(backend="template")
    evidence = [
        {
            "document_id": "9781513563602-mod01",
            "modality": "text",
            "text": "The model central bank maintained separate accounts for foreign currency and domestic currency reserves.",
        }
    ]
    sql_results = {
        "executed": True,
        "sql_query": "SELECT * FROM doc_table WHERE description LIKE '%assets%'",
        "sql_results": [{"description": "Total Assets", "2019": 5000, "2018": 4500}],
    }

    result = gen.generate(
        query="What was the total asset value as of December 31, 2019?",
        evidence=evidence,
        sql_results=sql_results,
    )

    assert "answer" in result
    assert "citations" in result
    assert result["engine"] == "offline_template_synthesizer"
    assert "description: Total Assets" in result["answer"]
    assert "| description | 2019 | 2018 |" in result["answer"]
    assert len(result["citations"]) >= 1


def test_generation_gpu_unavailable_fallback(monkeypatch):
    # Simulate GPU being unavailable
    monkeypatch.setattr(GenerationModule, "_is_gpu_available", staticmethod(lambda: False))
    gen = GenerationModule(backend="auto")

    evidence = [
        {
            "document_id": "doc_cpu_fallback",
            "modality": "text",
            "text": "Q3 net profit margin increased to 24.5% year-over-year.",
        }
    ]

    result = gen.generate(query="What was the Q3 profit margin?", evidence=evidence)

    assert "answer" in result
    assert result["fallback_used"] is True
    assert result["device"] in {"cpu", "cloud"}
    assert "24.5%" in result["answer"]
    assert len(result["citations"]) == 1


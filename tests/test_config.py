"""test_config.py — Unit & Integration Tests for DocuReason v1.1.0 Configuration Engine."""
from __future__ import annotations

import os
from pathlib import Path

from src.tripath.attribution.nli_attributor import NLIFaithfulnessAttributor
from src.tripath.config import DocuReasonConfig
from src.tripath.generation.generate import GenerationModule
from src.tripath.indexing.dense_index import DenseIndexBuilder
from src.tripath.retrieval.ranker import Ranker
from src.tripath.router.configurable_router import ConfigurableRouter


def test_preset_profiles_loading():
    """Verify that all 4 production profiles load correct hyperparameter settings."""
    quality_cfg = DocuReasonConfig.load_preset("quality_max")
    assert quality_cfg.indexing.index_type == "hnsw"
    assert quality_cfg.indexing.hnsw_m == 64
    assert quality_cfg.generation.max_context_tokens == 8192
    assert quality_cfg.attribution.enable_nli is True

    latency_cfg = DocuReasonConfig.load_preset("latency_optimized")
    assert latency_cfg.indexing.hnsw_m == 16
    assert latency_cfg.generation.max_context_tokens == 2048
    assert latency_cfg.attribution.enable_nli is False

    cpu_cfg = DocuReasonConfig.load_preset("low_resource_cpu")
    assert cpu_cfg.indexing.index_type == "flat"
    assert cpu_cfg.generation.backend == "template"
    assert cpu_cfg.attribution.enable_nli is False


def test_yaml_config_loading(tmp_path: Path):
    """Verify loading settings from config.yaml."""
    yaml_file = tmp_path / "custom_config.yaml"
    yaml_file.write_text(
        """
preset: "quality_max"
output_dir: "custom_artifacts"
indexing:
  domain: "biomedical"
  hnsw_m: 48
generation:
  max_context_tokens: 6000
""",
        encoding="utf-8",
    )

    cfg = DocuReasonConfig.load_from_yaml(yaml_file)
    assert cfg.output_dir == "custom_artifacts"
    assert cfg.indexing.domain == "biomedical"
    assert cfg.indexing.hnsw_m == 48
    assert cfg.generation.max_context_tokens == 6000


def test_environment_variable_overrides():
    """Verify environment variable overrides using DOCUREASON_*."""
    os.environ["DOCUREASON_INDEXING_DOMAIN"] = "financial"
    os.environ["DOCUREASON_GENERATION_MAX_CONTEXT_TOKENS"] = "12000"

    try:
        cfg = DocuReasonConfig.load_from_yaml("configs/config.yaml")
        assert cfg.indexing.domain == "financial"
        assert cfg.generation.max_context_tokens == 12000
    finally:
        os.environ.pop("DOCUREASON_INDEXING_DOMAIN", None)
        os.environ.pop("DOCUREASON_GENERATION_MAX_CONTEXT_TOKENS", None)


def test_configs_directory_preset_yaml_files():
    """Verify that preset YAML files inside configs/ directory load successfully."""
    quality_yaml = DocuReasonConfig.load_from_yaml("configs/quality_max.yaml")
    assert quality_yaml.preset == "quality_max"
    assert quality_yaml.indexing.hnsw_m == 64

    latency_yaml = DocuReasonConfig.load_from_yaml("configs/latency_optimized.yaml")
    assert latency_yaml.preset == "latency_optimized"
    assert latency_yaml.generation.max_context_tokens == 2048

    cpu_yaml = DocuReasonConfig.load_from_yaml("configs/low_resource_cpu.yaml")
    assert cpu_yaml.preset == "low_resource_cpu"
    assert cpu_yaml.generation.backend == "template"


def test_query_runtime_overrides():
    """Verify applying per-query runtime overrides."""
    cfg = DocuReasonConfig.load_preset("balanced")
    assert cfg.indexing.domain == "general"

    overrides = {"domain": "legal", "max_context_tokens": 16000, "final_top_k": 8}
    updated_cfg = cfg.apply_query_overrides(overrides)

    assert updated_cfg.indexing.domain == "legal"
    assert updated_cfg.generation.max_context_tokens == 16000
    assert updated_cfg.reranker.final_top_k == 8


def test_component_injection_with_config():
    """Verify component instantiation with custom DocuReasonConfig."""
    config = DocuReasonConfig.load_preset("quality_max")

    dense_builder = DenseIndexBuilder(config=config)
    assert dense_builder.domain == "biomedical" or dense_builder.index_type == "hnsw"
    assert dense_builder.hnsw_m == 64

    router = ConfigurableRouter(config=config)
    assert router.threshold == 0.35

    ranker = Ranker(config=config)
    assert ranker.short_heading_threshold == 45

    generator = GenerationModule(config=config)
    assert generator.max_context_tokens == 8192

    attributor = NLIFaithfulnessAttributor(config=config)
    assert attributor.enable_nli is True

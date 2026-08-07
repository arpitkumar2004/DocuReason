"""config.py — Centralized Hyperparameter Configuration Engine for DocuReason v1.1.0.

Provides structured, validated, and hierarchical configuration for all 7
RAG pipeline layers (Ingestion, Indexing, Routing, Retrieval, Reranking,
Generation, Attribution).

Supports:
1. Preset profiles: 'quality_max', 'balanced', 'latency_optimized', 'low_resource_cpu'.
2. YAML configuration file loading ('config.yaml').
3. Environment variable overrides ('DOCUREASON_*').
4. Per-query runtime overrides for REST API requests.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.tripath.utils import get_logger, log_pipeline_flag

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Layer Configurations
# ----------------------------------------------------------------------

@dataclass
class IngestionConfig:
    """Ingestion & Layout Parsing Parameters."""
    ocr_fallback_char_threshold: int = 50
    page_batch_size: int = 1
    image_rendering_dpi: int = 150
    enable_ocr_fallback: bool = True
    child_chunk_size: int = 256
    parent_region_size: int = 1024
    chunk_overlap: int = 32


@dataclass
class IndexingConfig:
    """Multi-Modal Indexing & FAISS HNSW Parameters."""
    domain: str = "general"
    model_name: Optional[str] = None
    index_type: str = "hnsw"  # "flat" or "hnsw"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class RouterConfig:
    """Intent Routing & Modality Keyword Parameters."""
    threshold: float = 0.35
    sigmoid_lambda: float = 1.2
    modality_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "text": ["revenue", "growth", "report", "overview", "company", "margin", "statement", "financial", "narrative"],
        "table": ["table", "region", "quarter", "by", "revenue", "sum", "average", "total", "rate", "percent"],
        "vision": ["chart", "figure", "image", "graph", "adoption", "bar", "pie", "diagram", "plot", "trend"],
    })


@dataclass
class RetrievalConfig:
    """Hybrid Retrieval & Late Fusion Parameters."""
    top_k_text: int = 20
    top_k_table: int = 20
    top_k_vision: int = 20
    rrf_k: int = 60
    enable_dynamic_router_weights: bool = True


@dataclass
class RerankerConfig:
    """Cross-Encoder Reranking & Parent-Child Expansion Parameters."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    short_heading_char_threshold: int = 45
    heading_penalty_multiplier: float = 0.5
    final_top_k: int = 5
    enable_parent_expansion: bool = True


@dataclass
class GenerationConfig:
    """Multimodal Generator & Context Budget Parameters."""
    backend: str = "auto"
    model_name_or_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    max_context_tokens: int = 4096
    truncation_strategy: str = "smart_relevance"
    temperature: float = 0.1
    max_new_tokens: int = 512


@dataclass
class AttributionConfig:
    """Sentence-Level NLI Faithfulness Attribution Parameters."""
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small"
    entailment_threshold: float = 0.5
    flag_threshold_precision: float = 0.5
    enable_nli: bool = True


# ----------------------------------------------------------------------
# Root Framework Configuration
# ----------------------------------------------------------------------

@dataclass
class DocuReasonConfig:
    """Central Master Configuration Object for DocuReason v1.1.0."""
    preset: str = "balanced"
    output_dir: str = "artifacts"
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a nested dict."""
        return asdict(self)

    @classmethod
    def load_preset(cls, preset_name: str) -> DocuReasonConfig:
        """Load one of the 4 pre-tuned production profile presets."""
        preset_clean = preset_name.lower().strip()
        config = cls(preset=preset_clean)

        if preset_clean == "quality_max":
            config.indexing.index_type = "hnsw"
            config.indexing.hnsw_m = 64
            config.indexing.hnsw_ef_construction = 400
            config.indexing.hnsw_ef_search = 128
            config.retrieval.top_k_text = 40
            config.retrieval.top_k_table = 40
            config.retrieval.top_k_vision = 40
            config.reranker.final_top_k = 10
            config.generation.max_context_tokens = 8192
            config.attribution.enable_nli = True

        elif preset_clean == "balanced":
            config.indexing.index_type = "hnsw"
            config.indexing.hnsw_m = 32
            config.indexing.hnsw_ef_construction = 200
            config.indexing.hnsw_ef_search = 64
            config.retrieval.top_k_text = 20
            config.retrieval.top_k_table = 20
            config.retrieval.top_k_vision = 20
            config.reranker.final_top_k = 5
            config.generation.max_context_tokens = 4096
            config.attribution.enable_nli = True

        elif preset_clean == "latency_optimized":
            config.indexing.index_type = "hnsw"
            config.indexing.hnsw_m = 16
            config.indexing.hnsw_ef_construction = 100
            config.indexing.hnsw_ef_search = 32
            config.retrieval.top_k_text = 10
            config.retrieval.top_k_table = 10
            config.retrieval.top_k_vision = 10
            config.reranker.final_top_k = 3
            config.generation.max_context_tokens = 2048
            config.attribution.enable_nli = False

        elif preset_clean == "low_resource_cpu":
            config.indexing.index_type = "flat"
            config.indexing.domain = "general"
            config.retrieval.top_k_text = 10
            config.retrieval.top_k_table = 10
            config.retrieval.top_k_vision = 10
            config.reranker.final_top_k = 3
            config.generation.backend = "template"
            config.generation.max_context_tokens = 1500
            config.attribution.enable_nli = False

        else:
            logger.warning("Unknown preset '%s' — using default 'balanced'", preset_name)

        log_pipeline_flag("config_preset_loaded", preset_clean, "Centralized configuration preset", logger)
        return config

    @classmethod
    def load_from_yaml(cls, yaml_path: Union[str, Path] = "configs/config.yaml") -> DocuReasonConfig:
        """Load configuration from a YAML file in the `configs/` directory with env variable overrides."""
        path = Path(yaml_path)
        if not path.exists():
            # Fallback search locations
            candidates = [Path("configs/config.yaml"), Path("configs/default.yaml"), Path("config.yaml")]
            for cand in candidates:
                if cand.exists():
                    path = cand
                    break

        data: Dict[str, Any] = {}
        if path.exists():
            try:
                import yaml
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                logger.info("Loaded DocuReasonConfig from YAML: %s", path)
            except Exception as exc:
                logger.warning("Failed to parse YAML config %s (%s) — falling back to baseline defaults", path, exc)
        else:
            logger.info("No YAML config file found at %s — using default preset profile", yaml_path)

        preset = data.get("preset", "balanced")
        config = cls.load_preset(preset)

        # Merge YAML sections into dataclasses if present
        if "output_dir" in data:
            config.output_dir = data["output_dir"]

        for section_name in ["ingestion", "indexing", "router", "retrieval", "reranker", "generation", "attribution"]:
            if section_name in data and isinstance(data[section_name], dict):
                section_obj = getattr(config, section_name)
                for k, v in data[section_name].items():
                    if hasattr(section_obj, k):
                        setattr(section_obj, k, v)

        # Environment variable overrides (DOCUREASON_SECTION_FIELD or DOCUREASON_FIELD)
        sections = ["ingestion", "indexing", "router", "retrieval", "reranker", "generation", "attribution"]
        for env_key, env_val in os.environ.items():
            if env_key.startswith("DOCUREASON_"):
                raw = env_key[len("DOCUREASON_"):].lower()
                matched = False
                for sec_name in sections:
                    if raw.startswith(f"{sec_name}_"):
                        field_name = raw[len(sec_name) + 1:]
                        sec_obj = getattr(config, sec_name)
                        if hasattr(sec_obj, field_name):
                            field_type = type(getattr(sec_obj, field_name))
                            try:
                                if field_type is bool:
                                    cast_val = env_val.lower() in ("1", "true", "yes")
                                else:
                                    cast_val = field_type(env_val)
                                setattr(sec_obj, field_name, cast_val)
                                logger.info("Env override applied: %s -> %s", env_key, cast_val)
                                matched = True
                                break
                            except Exception as exc:
                                logger.warning("Failed to cast env override %s: %s", env_key, exc)
                if not matched:
                    # Try direct field match across sections
                    for sec_name in sections:
                        sec_obj = getattr(config, sec_name)
                        if hasattr(sec_obj, raw):
                            field_type = type(getattr(sec_obj, raw))
                            try:
                                cast_val = env_val.lower() in ("1", "true", "yes") if field_type is bool else field_type(env_val)
                                setattr(sec_obj, raw, cast_val)
                                logger.info("Env override applied: %s -> %s", env_key, cast_val)
                                break
                            except Exception:
                                pass

        return config

    def apply_query_overrides(self, overrides: Dict[str, Any]) -> DocuReasonConfig:
        """Apply dynamic runtime per-query overrides (for FastAPI JSON requests)."""
        if not overrides:
            return self

        # Direct shortcuts (e.g. {"domain": "legal", "max_context_tokens": 8192, "final_top_k": 5})
        if "domain" in overrides:
            self.indexing.domain = str(overrides["domain"])
        if "max_context_tokens" in overrides:
            self.generation.max_context_tokens = int(overrides["max_context_tokens"])
        if "final_top_k" in overrides:
            self.reranker.final_top_k = int(overrides["final_top_k"])
        if "index_type" in overrides:
            self.indexing.index_type = str(overrides["index_type"])
        if "gen_backend" in overrides:
            self.generation.backend = str(overrides["gen_backend"])

        return self

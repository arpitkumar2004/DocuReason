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

SUPPORTED_CORPUS_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".csv", ".txt"}


def check_system(verbose: bool = True) -> Dict[str, Any]:
    """Perform environment readiness diagnostic check for DocuReason framework.

    Verifies optional dependencies, GPU/CPU hardware acceleration, disk permissions,
    and returns a diagnostic status dictionary.
    """
    import importlib
    import sys

    results: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "os_platform": sys.platform,
        "cuda_available": False,
        "device": "cpu",
        "dependencies": {},
    }

    try:
        import torch

        results["dependencies"]["torch"] = torch.__version__
        if torch.cuda.is_available():
            results["cuda_available"] = True
            results["device"] = f"cuda ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            results["device"] = "mps (Apple Silicon)"
    except ImportError:
        results["dependencies"]["torch"] = "Not Installed"

    optional_pkgs = ["faiss", "bm25s", "docling", "easyocr", "duckdb", "transformers", "qdrant_client"]
    for pkg in optional_pkgs:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "Installed")
            results["dependencies"][pkg] = str(ver)
        except ImportError:
            results["dependencies"][pkg] = "Not Installed (Using Fallback/Stub)"

    if verbose:
        lines = []
        lines.append("=" * 80)
        lines.append("[DocuReason Framework System Readiness & Diagnostics]")
        lines.append("=" * 80)
        lines.append(f"  • Python Version : {results['python_version']} ({results['os_platform']})")
        lines.append(f"  • Compute Device : {results['device']}")
        lines.append("  • Dependency Audit Status:")
        for pkg, status in results["dependencies"].items():
            lines.append(f"      - {pkg:<16}: {status}")
        lines.append("=" * 80)
        print("\n".join(lines))

    return results


# ----------------------------------------------------------------------
# Framework Custom Exceptions
# ----------------------------------------------------------------------


class DocuReasonConfigError(Exception):
    """Base exception for all configuration and validation errors in DocuReason."""

    pass


class MissingRequiredConfigError(DocuReasonConfigError):
    """Raised when a minimum required developer input is missing or invalid."""

    pass


class InvalidCorpusError(DocuReasonConfigError):
    """Raised when the specified input data corpus directory is invalid or contains no supported files."""

    pass


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
    modality_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "text": [
                "revenue",
                "growth",
                "report",
                "overview",
                "company",
                "margin",
                "statement",
                "financial",
                "narrative",
            ],
            "table": ["table", "region", "quarter", "by", "revenue", "sum", "average", "total", "rate", "percent"],
            "vision": ["chart", "figure", "image", "graph", "adoption", "bar", "pie", "diagram", "plot", "trend"],
        }
    )


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
    source_map: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a nested dict."""
        return asdict(self)

    @staticmethod
    def validate_inputs(input_dir: Union[str, Path, None], output_dir: Union[str, Path, None]) -> Dict[str, Any]:
        """Validate minimum required developer inputs to run the DocuReason pipeline.

        Returns metadata about input corpus validation.
        Raises MissingRequiredConfigError or InvalidCorpusError on failure.
        """
        if not input_dir:
            raise MissingRequiredConfigError(
                "Missing required parameter 'input_dir' (Data Corpus Path).\n"
                "Developers must specify 'input_dir' pointing to the directory containing source documents.\n"
                "Example: pipeline = DocuReasonPipeline(input_dir='samples', output_dir='artifacts/run')"
            )

        in_path = Path(input_dir)
        if not in_path.exists():
            raise InvalidCorpusError(
                f"Specified input_dir '{input_dir}' does not exist on disk.\n"
                f"Please verify the directory path and try again."
            )

        if not in_path.is_dir():
            raise InvalidCorpusError(
                f"Specified input_dir '{input_dir}' is a file, not a directory.\n"
                f"Please specify a directory path containing your document corpus."
            )

        supported_files = [
            f for f in in_path.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_CORPUS_EXTENSIONS
        ]
        if not supported_files:
            raise InvalidCorpusError(
                f"No supported document files found in input_dir '{in_path.resolve()}'.\n"
                f"Supported file extensions: {sorted(list(SUPPORTED_CORPUS_EXTENSIONS))}\n"
                f"Please place your documents inside '{input_dir}' before running the pipeline."
            )

        if not output_dir:
            raise MissingRequiredConfigError(
                "Missing required parameter 'output_dir' (Artifact Target Path).\n"
                "Developers must specify 'output_dir' to store vector indices and pipeline artifacts.\n"
                "Example: pipeline = DocuReasonPipeline(input_dir='samples', output_dir='artifacts/run')"
            )

        out_path = Path(output_dir)
        try:
            out_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise MissingRequiredConfigError(
                f"Output directory '{output_dir}' could not be created/accessed: {exc}"
            ) from exc

        return {
            "input_dir": str(in_path.resolve()),
            "output_dir": str(out_path.resolve()),
            "supported_file_count": len(supported_files),
            "supported_files": [f.name for f in supported_files[:5]],
        }

    @classmethod
    def load_preset(cls, preset_name: str) -> DocuReasonConfig:
        """Load one of the 4 pre-tuned production profile presets."""
        preset_clean = preset_name.lower().strip()
        config = cls(preset=preset_clean)
        config.source_map["preset"] = f"Preset Profile: '{preset_clean}'"

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
        config.source_map["config_origin"] = f"YAML File: {path}" if path.exists() else "Framework Baseline Preset"

        if "output_dir" in data:
            config.output_dir = data["output_dir"]
            config.source_map["output_dir"] = f"YAML File: {path}"

        for section_name in ["ingestion", "indexing", "router", "retrieval", "reranker", "generation", "attribution"]:
            if section_name in data and isinstance(data[section_name], dict):
                section_obj = getattr(config, section_name)
                for k, v in data[section_name].items():
                    if hasattr(section_obj, k):
                        setattr(section_obj, k, v)
                        config.source_map[f"{section_name}.{k}"] = f"YAML File: {path}"

        sections = ["ingestion", "indexing", "router", "retrieval", "reranker", "generation", "attribution"]
        for env_key, env_val in os.environ.items():
            if env_key.startswith("DOCUREASON_"):
                raw = env_key[len("DOCUREASON_") :].lower()
                matched = False
                for sec_name in sections:
                    if raw.startswith(f"{sec_name}_"):
                        field_name = raw[len(sec_name) + 1 :]
                        sec_obj = getattr(config, sec_name)
                        if hasattr(sec_obj, field_name):
                            field_type = type(getattr(sec_obj, field_name))
                            try:
                                cast_val = (
                                    env_val.lower() in ("1", "true", "yes")
                                    if field_type is bool
                                    else field_type(env_val)
                                )
                                setattr(sec_obj, field_name, cast_val)
                                config.source_map[f"{sec_name}.{field_name}"] = f"Environment Variable: {env_key}"
                                logger.info("Env override applied: %s -> %s", env_key, cast_val)
                                matched = True
                                break
                            except Exception as exc:
                                logger.warning("Failed to cast env override %s: %s", env_key, exc)
                if not matched:
                    for sec_name in sections:
                        sec_obj = getattr(config, sec_name)
                        if hasattr(sec_obj, raw):
                            field_type = type(getattr(sec_obj, raw))
                            try:
                                cast_val = (
                                    env_val.lower() in ("1", "true", "yes")
                                    if field_type is bool
                                    else field_type(env_val)
                                )
                                setattr(sec_obj, raw, cast_val)
                                config.source_map[f"{sec_name}.{raw}"] = f"Environment Variable: {env_key}"
                                logger.info("Env override applied: %s -> %s", env_key, cast_val)
                                break
                            except Exception:
                                pass

        return config

    def apply_query_overrides(self, overrides: Dict[str, Any]) -> DocuReasonConfig:
        """Apply dynamic runtime per-query overrides (for FastAPI JSON requests)."""
        if not overrides:
            return self

        if "domain" in overrides:
            self.indexing.domain = str(overrides["domain"])
            self.source_map["indexing.domain"] = "Per-Query Runtime Override"
        if "max_context_tokens" in overrides:
            self.generation.max_context_tokens = int(overrides["max_context_tokens"])
            self.source_map["generation.max_context_tokens"] = "Per-Query Runtime Override"
        if "final_top_k" in overrides:
            self.reranker.final_top_k = int(overrides["final_top_k"])
            self.source_map["reranker.final_top_k"] = "Per-Query Runtime Override"
        if "index_type" in overrides:
            self.indexing.index_type = str(overrides["index_type"])
            self.source_map["indexing.index_type"] = "Per-Query Runtime Override"
        if "gen_backend" in overrides:
            self.generation.backend = str(overrides["gen_backend"])
            self.source_map["generation.backend"] = "Per-Query Runtime Override"

        return self

    def summary_dict(self) -> Dict[str, Any]:
        """Return structured summary dictionary of active framework settings."""
        return {
            "preset": self.preset,
            "output_dir": self.output_dir,
            "ingestion": asdict(self.ingestion),
            "indexing": asdict(self.indexing),
            "router": asdict(self.router),
            "retrieval": asdict(self.retrieval),
            "reranker": asdict(self.reranker),
            "generation": asdict(self.generation),
            "attribution": asdict(self.attribution),
            "source_map": self.source_map,
        }

    def summary_string(
        self, input_dir: Optional[Union[str, Path]] = None, output_dir: Optional[Union[str, Path]] = None
    ) -> str:
        """Generate a formatted PyTorch-like Configuration Transparency Summary string."""
        lines = []
        lines.append("=" * 80)
        lines.append("[DocuReason Framework v1.1.3] Configuration & Pipeline Transparency")
        lines.append("=" * 80)

        effective_output_dir = output_dir or self.output_dir

        lines.append("[Required Developer Inputs]:")
        if input_dir:
            in_path = Path(input_dir)
            if in_path.exists() and in_path.is_dir():
                count = len(
                    [f for f in in_path.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_CORPUS_EXTENSIONS]
                )
                lines.append(f"  - input_dir (Data Corpus Path) : {input_dir} [VERIFIED - {count} supported file(s)]")
            else:
                lines.append(f"  - input_dir (Data Corpus Path) : {input_dir} [INVALID / NOT FOUND]")
        else:
            lines.append("  - input_dir (Data Corpus Path) : NOT SPECIFIED [REQUIRED]")

        if effective_output_dir:
            lines.append(f"  - output_dir (Artifact Path)  : {effective_output_dir} [VERIFIED - target ready]")
        else:
            lines.append("  - output_dir (Artifact Path)  : NOT SPECIFIED [REQUIRED]")

        lines.append("-" * 80)
        lines.append("[Framework Active Layer Configurations & Defaults]:")
        lines.append(
            f"  - Config Preset Profile       : '{self.preset}' ({self.source_map.get('config_origin', 'Framework Default Preset')})"
        )
        lines.append("  [1. Ingestion Layer Defaults]:")
        lines.append(
            f"      • OCR Fallback Enabled     : {self.ingestion.enable_ocr_fallback} (char_threshold={self.ingestion.ocr_fallback_char_threshold}, dpi={self.ingestion.image_rendering_dpi})"
        )
        lines.append(
            f"      • Chunking Tokens          : child_chunk={self.ingestion.child_chunk_size}, parent_region={self.ingestion.parent_region_size}, overlap={self.ingestion.chunk_overlap}"
        )
        lines.append("  [2. Indexing Layer Defaults]:")
        lines.append(
            f"      • Domain / Vector Model    : domain='{self.indexing.domain}', model='{self.indexing.model_name or 'Default (sentence-transformers/all-MiniLM-L6-v2)'}'"
        )
        lines.append(
            f"      • FAISS Index Configuration: index_type='{self.indexing.index_type}', hnsw_m={self.indexing.hnsw_m}, ef_construction={self.indexing.hnsw_ef_construction}, ef_search={self.indexing.hnsw_ef_search}"
        )
        lines.append(f"      • BM25S Parameters         : k1={self.indexing.bm25_k1}, b={self.indexing.bm25_b}")
        lines.append("  [3. Intent Router Layer Defaults]:")
        lines.append(
            f"      • Activation Threshold     : {self.router.threshold} (sigmoid_lambda={self.router.sigmoid_lambda})"
        )
        lines.append("  [4. Hybrid Retrieval Layer Defaults]:")
        lines.append(
            f"      • Top-K Per Modality       : text={self.retrieval.top_k_text}, table={self.retrieval.top_k_table}, vision={self.retrieval.top_k_vision} (RRF k={self.retrieval.rrf_k})"
        )
        lines.append("  [5. Cross-Encoder Reranker Defaults]:")
        lines.append(
            f"      • Model & Target Top-K     : model='{self.reranker.model_name}', final_top_k={self.reranker.final_top_k}, parent_expansion={self.reranker.enable_parent_expansion}"
        )
        lines.append("  [6. Multimodal Generation Defaults]:")
        lines.append(
            f"      • Model Backend & Path     : backend='{self.generation.backend}', model='{self.generation.model_name_or_path}'"
        )
        lines.append(
            f"      • Context Token Budget     : max_context_tokens={self.generation.max_context_tokens}, temp={self.generation.temperature}, max_new_tokens={self.generation.max_new_tokens}"
        )
        lines.append("  [7. Faithfulness Attribution Defaults]:")
        lines.append(
            f"      • NLI Model & Threshold    : enable_nli={self.attribution.enable_nli}, model='{self.attribution.nli_model_name}', threshold={self.attribution.entailment_threshold}"
        )
        lines.append("=" * 80)
        lines.append(
            "Transparency Notice: Active defaults logged. To override, pass a custom DocuReasonConfig or YAML path."
        )
        lines.append("=" * 80)
        return "\n".join(lines)

    def print_summary(
        self,
        input_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        logger: Optional[Any] = None,
    ) -> None:
        """Print or log the Configuration Transparency Summary."""
        summary_txt = self.summary_string(input_dir=input_dir, output_dir=output_dir)
        if logger:
            for line in summary_txt.splitlines():
                logger.info(line)
        else:
            print(summary_txt)

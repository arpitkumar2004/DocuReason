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
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from docureason.tripath.utils import get_logger, log_pipeline_flag
from docureason.tripath.version import __version__

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


_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})
_NONE_TOKENS = frozenset({"", "none", "null", "auto", "default"})


def _cast_env_value(section: Any, field_name: str, raw_value: str) -> Any:
    """Cast an environment-variable string to a dataclass field's declared type.

    The caster is driven by the field's *annotation*, not by the runtime type of
    its current value.  ``type(getattr(...))`` was used before, which resolved to
    ``NoneType`` for every ``Optional[...]`` field left at its ``None`` default —
    so ``DOCUREASON_INDEXING_MODEL_NAME`` and every other optional field silently
    failed to apply.  ``Optional`` fields also accept an explicit
    ``""``/``none``/``auto`` to mean "unset".
    """
    annotation = ""
    for f in dataclass_fields(section):
        if f.name == field_name:
            annotation = str(f.type)
            break

    optional = "Optional" in annotation or "None" in annotation
    if optional and raw_value.strip().lower() in _NONE_TOKENS:
        return None

    token = raw_value.strip().lower()
    if "bool" in annotation:
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
        raise ValueError(f"{field_name}: expected a boolean, got {raw_value!r}")
    if "int" in annotation:
        return int(raw_value)
    if "float" in annotation:
        return float(raw_value)
    if "str" in annotation:
        return raw_value
    # Dict/List fields are configured through YAML only.
    raise TypeError(f"{field_name}: type {annotation!r} is not settable from the environment")


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
    # Tri-state TableFormer table-structure recognition.  None = auto (on when
    # CUDA is available).  Previously this was decided by a bare CUDA check plus
    # two undocumented environment variables, so CPU users silently got degraded
    # table parsing while the docs claimed TableFormer had run (audit P0-6).
    enable_table_structure: Optional[bool] = None
    # Ingestion-time ablation switch for the Docling layout parse (ablation A1).
    # False forces the rule-based line segmenter that used to be reachable only by
    # making Docling fail, so "no layout parser" can be measured rather than assumed.
    enable_layout_parser: bool = True


@dataclass
class IndexingConfig:
    """Multi-Modal Indexing & FAISS HNSW Parameters."""

    domain: str = "general"
    model_name: Optional[str] = None
    # Pinned HuggingFace revision (commit SHA, tag or branch) for the embedding
    # model.  None follows the hub default, which means the weights behind a run
    # can change without any repository change (audit S7 / P3-4).
    model_revision: Optional[str] = None
    index_type: str = "hnsw"  # "flat" or "hnsw"
    # Below this many vectors a flat index is both exact and faster than an HNSW
    # graph, so index_type="hnsw" resolves to flat.  The committed artifacts held
    # 193 vectors, where the graph cost build time and recall for nothing.
    hnsw_min_vectors: int = 10000
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
    # When True the retrievers query the FAISS + BM25S indexes built at ingestion
    # time and fall back to the in-memory substring scan only if no index is
    # readable.  Set False to force the legacy scan (used by the retrieval
    # ablation and by tests that must not touch an index).
    enable_index_retrieval: bool = True
    # Query-time ablation switches. Each names a retrieval path that the ablation
    # matrix in Documentations/eval_method.md claims a delta for; without a switch
    # those deltas could only be asserted, never measured.
    enable_sparse_retrieval: bool = True
    enable_table_sql: bool = True


@dataclass
class RerankerConfig:
    """Cross-Encoder Reranking & Parent-Child Expansion Parameters."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    #: Pinned HuggingFace revision for the cross-encoder; None follows the hub default.
    model_revision: Optional[str] = None
    short_heading_char_threshold: int = 45
    # Subtractive penalty applied to standalone short headings when the
    # cross-encoder is active.  Cross-encoder outputs are *unbounded signed
    # logits*, so a multiplicative penalty (< 1.0) moves negative scores
    # toward zero and therefore PROMOTES low-scoring headings.  Subtraction is
    # monotone decreasing for every sign and is the correct operator here.
    heading_penalty_offset: float = 2.0
    # Retained for the non-negative fallback scorer (Jaccard path) and for
    # backward compatibility with configs written before v1.1.4.
    heading_penalty_multiplier: float = 0.5
    final_top_k: int = 5
    enable_parent_expansion: bool = True
    # False forces the non-model Jaccard fallback scorer, which is what a
    # "no cross-encoder" ablation means (ablation A3).
    enable_cross_encoder: bool = True
    # False disables the short-heading penalty entirely (ablation A5). Distinct from
    # setting the offset to 0.0, which would still record the penalty as active.
    enable_heading_penalty: bool = True


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
class ServingConfig:
    """Online serving behaviour: concurrency and caching.

    ``configs/serving.yaml`` existed but was never loaded (``load_from_yaml`` reads
    ``config.yaml`` only), so both of these were previously unreachable settings for
    behaviour the service did not have.
    """

    #: Distinct (query, config) responses kept in the per-service LRU. 0 disables it.
    response_cache_size: int = 64
    #: Run the three modality retrievers concurrently instead of one after another.
    #: Results are collected in a fixed order, so this changes latency, not ranking.
    enable_parallel_paths: bool = True


@dataclass
class AttributionConfig:
    """Sentence-Level NLI Faithfulness Attribution Parameters."""

    nli_model_name: str = "cross-encoder/nli-deberta-v3-small"
    entailment_threshold: float = 0.5
    flag_threshold_precision: float = 0.5
    enable_nli: bool = True


#: Every configurable layer, in banner order.  One tuple, so adding a section does
#: not mean remembering to add it to the YAML loader, the env-override scan and the
#: summary independently.
_CONFIG_SECTIONS = (
    "ingestion",
    "indexing",
    "router",
    "retrieval",
    "reranker",
    "generation",
    "attribution",
    "serving",
)


def _as_bool(value: Any) -> bool:
    """Coerce a JSON or CLI value to bool, rejecting anything ambiguous."""
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


#: Per-query override name -> (section, field, caster).  One table, so a field is
#: addressable from a REST request body and from an ablation definition by the same
#: name; adding an ablation switch means adding one row here.
_QUERY_OVERRIDES: Dict[str, tuple] = {
    "domain": ("indexing", "domain", str),
    "index_type": ("indexing", "index_type", str),
    "max_context_tokens": ("generation", "max_context_tokens", int),
    "gen_backend": ("generation", "backend", str),
    "final_top_k": ("reranker", "final_top_k", int),
    "top_k_text": ("retrieval", "top_k_text", int),
    "top_k_table": ("retrieval", "top_k_table", int),
    "top_k_vision": ("retrieval", "top_k_vision", int),
    # Ablation switches (Documentations/eval_method.md §21).
    "enable_index_retrieval": ("retrieval", "enable_index_retrieval", _as_bool),
    "enable_sparse_retrieval": ("retrieval", "enable_sparse_retrieval", _as_bool),
    "enable_table_sql": ("retrieval", "enable_table_sql", _as_bool),
    "enable_cross_encoder": ("reranker", "enable_cross_encoder", _as_bool),
    "enable_heading_penalty": ("reranker", "enable_heading_penalty", _as_bool),
    "enable_parent_expansion": ("reranker", "enable_parent_expansion", _as_bool),
    "enable_nli": ("attribution", "enable_nli", _as_bool),
    "enable_parallel_paths": ("serving", "enable_parallel_paths", _as_bool),
    "response_cache_size": ("serving", "response_cache_size", int),
}


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
    serving: ServingConfig = field(default_factory=ServingConfig)
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

        for section_name in _CONFIG_SECTIONS:
            if section_name in data and isinstance(data[section_name], dict):
                section_obj = getattr(config, section_name)
                for k, v in data[section_name].items():
                    if hasattr(section_obj, k):
                        setattr(section_obj, k, v)
                        config.source_map[f"{section_name}.{k}"] = f"YAML File: {path}"

        sections = list(_CONFIG_SECTIONS)
        for env_key, env_val in os.environ.items():
            if not env_key.startswith("DOCUREASON_"):
                continue
            raw = env_key[len("DOCUREASON_") :].lower()

            # Prefer a fully qualified "<section>_<field>" name, then fall back to
            # a bare field name resolved against each section in order.
            targets = []
            for sec_name in sections:
                if raw.startswith(f"{sec_name}_"):
                    targets.append((sec_name, raw[len(sec_name) + 1 :]))
            targets.extend((sec_name, raw) for sec_name in sections)

            for sec_name, field_name in targets:
                sec_obj = getattr(config, sec_name)
                if not hasattr(sec_obj, field_name):
                    continue
                try:
                    cast_val = _cast_env_value(sec_obj, field_name, env_val)
                except (TypeError, ValueError) as exc:
                    logger.warning("Failed to cast env override %s=%r: %s", env_key, env_val, exc)
                    break
                setattr(sec_obj, field_name, cast_val)
                config.source_map[f"{sec_name}.{field_name}"] = f"Environment Variable: {env_key}"
                logger.info("Env override applied: %s -> %s", env_key, cast_val)
                break

        return config

    def apply_query_overrides(self, overrides: Dict[str, Any]) -> DocuReasonConfig:
        """Return a **copy** of this config with per-query overrides applied.

        This used to mutate ``self`` and return it, so a single request carrying
        ``{"final_top_k": 1}`` permanently narrowed the shared ``QueryService`` for
        every request that followed, and an ablation run would leave its own switches
        behind for the next one.  Overrides are per query by definition; deep-copying
        is what makes that true.

        Unknown keys are ignored but recorded in ``source_map`` under
        ``"unknown_overrides"``, so a typo in a request body is visible rather than
        silently doing nothing.
        """
        if not overrides:
            return self

        clone = deepcopy(self)
        unknown: List[str] = []

        for key, raw in overrides.items():
            spec = _QUERY_OVERRIDES.get(key)
            if spec is None:
                unknown.append(str(key))
                continue
            section_name, field_name, caster = spec
            section = getattr(clone, section_name)
            try:
                setattr(section, field_name, caster(raw))
            except (TypeError, ValueError) as exc:
                logger.warning("Ignoring per-query override %s=%r: %s", key, raw, exc)
                unknown.append(str(key))
                continue
            clone.source_map[f"{section_name}.{field_name}"] = "Per-Query Runtime Override"

        if unknown:
            clone.source_map["unknown_overrides"] = ", ".join(sorted(unknown))
            logger.warning("Per-query overrides ignored (unknown or invalid): %s", sorted(unknown))

        return clone

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
            "serving": asdict(self.serving),
            "source_map": self.source_map,
        }

    def summary_string(
        self, input_dir: Optional[Union[str, Path]] = None, output_dir: Optional[Union[str, Path]] = None
    ) -> str:
        """Generate a formatted PyTorch-like Configuration Transparency Summary string."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"[DocuReason Framework v{__version__}] Configuration & Pipeline Transparency")
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

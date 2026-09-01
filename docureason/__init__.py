"""DocuReason package for enterprise-grade Tri-Path Multimodal RAG."""

from docureason.ingestion import DocuReasonPipeline, Phase1Pipeline
from docureason.serving import QueryService
from docureason.tripath.attribution.claim_attributor import ClaimSupportAttributor, NLIFaithfulnessAttributor
from docureason.tripath.config import (
    DocuReasonConfig,
    DocuReasonConfigError,
    InvalidCorpusError,
    MissingRequiredConfigError,
    check_system,
)
from docureason.tripath.evaluation.artifact_quality import ArtifactQualityAuditor
from docureason.tripath.evaluation.dataset_exporter import DatasetExporter
from docureason.tripath.evaluation.eval_harness import EvaluationHarness, resolve_gold_ids
from docureason.tripath.evaluation.table_eval import TableEvaluator
from docureason.tripath.version import __version__

__all__ = [
    "DocuReasonPipeline",
    "Phase1Pipeline",
    "QueryService",
    "DocuReasonConfig",
    "DocuReasonConfigError",
    "MissingRequiredConfigError",
    "InvalidCorpusError",
    "check_system",
    "EvaluationHarness",
    "resolve_gold_ids",
    "TableEvaluator",
    "ArtifactQualityAuditor",
    "DatasetExporter",
    "ClaimSupportAttributor",
    # Deprecated alias, retained for backward compatibility.
    "NLIFaithfulnessAttributor",
    "__version__",
]

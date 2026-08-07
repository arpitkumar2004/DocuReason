"""DocuReason package for enterprise-grade Tri-Path Multimodal RAG."""

from docureason.ingestion import DocuReasonPipeline, Phase1Pipeline
from docureason.serving import QueryService
from src.tripath.attribution.nli_attributor import NLIFaithfulnessAttributor
from src.tripath.evaluation.artifact_quality import ArtifactQualityAuditor
from src.tripath.evaluation.dataset_exporter import DatasetExporter
from src.tripath.evaluation.eval_harness import EvaluationHarness
from src.tripath.evaluation.table_eval import TableEvaluator

__version__ = "1.1.1"

__all__ = [
    "DocuReasonPipeline",
    "Phase1Pipeline",
    "QueryService",
    "EvaluationHarness",
    "TableEvaluator",
    "ArtifactQualityAuditor",
    "DatasetExporter",
    "NLIFaithfulnessAttributor",
    "__version__",
]

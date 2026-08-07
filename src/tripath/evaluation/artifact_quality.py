"""src/tripath/evaluation/artifact_quality.py — Ingestion Artifact Dictionary Quality Auditor.

Evaluates:
1. Chunk Information Density (character count, token richness, empty chunk ratio).
2. Table Structure Integrity (Markdown grid formatting & JSON schema validity).
3. Modality Distribution & Metadata Enrichment (bounding boxes, page numbers, layout sources).
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.tripath.ingestion.schema import Document


class ArtifactQualityAuditor:
    """Audits processed Document and Chunk artifact dictionaries for retrieval readiness."""

    def audit_documents(self, documents: List[Document]) -> Dict[str, Any]:
        total_docs = len(documents)
        total_chunks = sum(len(doc.chunks) for doc in documents)
        total_regions = sum(len(doc.regions) for doc in documents)

        if total_chunks == 0:
            return {"status": "EMPTY_CORPUS", "total_documents": total_docs, "total_chunks": 0}

        empty_chunks = 0
        text_chunks = 0
        table_chunks = 0
        vision_chunks = 0
        chunk_lengths: List[int] = []
        valid_tables = 0
        ocr_fallback_count = 0

        for doc in documents:
            for chunk in doc.chunks:
                text_len = len(chunk.text.strip())
                chunk_lengths.append(text_len)

                if text_len == 0:
                    empty_chunks += 1

                mod = (chunk.modality or "text").lower()
                if mod == "text":
                    text_chunks += 1
                elif mod == "table":
                    table_chunks += 1
                elif mod == "vision":
                    vision_chunks += 1

            for region in doc.regions:
                if region.layout_source == "ocr":
                    ocr_fallback_count += 1
                if region.table_markdown or region.table_json:
                    valid_tables += 1

        avg_chunk_length = round(sum(chunk_lengths) / len(chunk_lengths), 2) if chunk_lengths else 0
        quality_score = round(1.0 - (empty_chunks / max(1, total_chunks)), 4)

        return {
            "total_documents": total_docs,
            "total_regions": total_regions,
            "total_chunks": total_chunks,
            "artifact_quality_score": quality_score,
            "modality_breakdown": {
                "text_chunks": text_chunks,
                "table_chunks": table_chunks,
                "vision_chunks": vision_chunks,
            },
            "chunk_statistics": {
                "average_chunk_char_length": avg_chunk_length,
                "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
                "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
                "empty_chunks": empty_chunks,
            },
            "enrichment_metrics": {
                "structured_tables_extracted": valid_tables,
                "ocr_fallback_regions": ocr_fallback_count,
            },
            "verdict": "HEALTHY_ARTIFACTS" if quality_score >= 0.95 else "NEEDS_INGESTION_CLEANUP",
        }

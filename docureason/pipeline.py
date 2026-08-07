from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tripath.indexing.artifact_writer import ArtifactWriter
from src.tripath.indexing.dense_index import DenseIndexBuilder
from src.tripath.indexing.sparse_index import BM25SIndexBuilder
from src.tripath.ingestion.docling_layout_parser import DoclingLayoutParser
from src.tripath.ingestion.figure_captioner import FigureCaptioner
from src.tripath.ingestion.format_loader import FormatAwareLoader
from src.tripath.ingestion.identity import IdentityManager
from src.tripath.ingestion.ocr_fallback import OCRFallback
from src.tripath.ingestion.schema import Region
from src.tripath.ingestion.table_serializer import TableSerializer
from src.tripath.utils import (
    log_pipeline_flag,
    setup_logger,
    trace_execution,
    trace_pipeline_stage,
)

logger = setup_logger("docureason.pipeline")


class DocuReasonPipeline:
    """Build a corpus and index artifact from documents using vision-based layout parsing.

    Processing order for each document
    ------------------------------------
    1. **Deep Docling parse** (``DoclingLayoutParser``) — layout-aware region
       classification using TableFormer + DocLayNet.  Falls back to the fast
       text-only path when Docling is unavailable.
    2. **OCR fallback** (``OCRFallback``) — scanned/image-only pages are
       detected by per-page character count and re-processed with EasyOCR.
    3. **Table enrichment** (``TableSerializer``) — every ``table`` region is
       converted to GFM Markdown, HTML, and a JSON schema
       ``{"columns": [...], "rows": [[...]]}`` for downstream SQL generation.
    """

    def __init__(self, input_dir: str | Path, output_dir: str | Path) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initializing DocuReasonPipeline: input_dir=%s, output_dir=%s", self.input_dir, self.output_dir)

        self.identity = IdentityManager()
        self.artifact_writer = ArtifactWriter(self.output_dir)
        self.loader = FormatAwareLoader()

        self.layout_parser = DoclingLayoutParser(page_batch_size=1, do_ocr=False)
        log_pipeline_flag("layout_parser_batch_size", 1, "Docling single-page memory limit", logger)

        self.ocr_fallback = OCRFallback(languages=["en"])
        log_pipeline_flag("ocr_languages", ["en"], "EasyOCR fallback configuration", logger)

        self.table_serializer = TableSerializer()

        self.figure_captioner = FigureCaptioner(use_blip2=True, use_clip=True)
        log_pipeline_flag("use_blip2", True, "BLIP-2 figure captioning enabled", logger)
        log_pipeline_flag("use_clip", True, "CLIP figure embeddings enabled", logger)

        self.dense_index = DenseIndexBuilder(output_dir=self.output_dir)
        self.sparse_index = BM25SIndexBuilder(output_dir=self.output_dir)

    @trace_pipeline_stage("Document Ingestion & Indexing Pipeline")
    def run(self) -> Dict[str, object]:
        documents: List[Dict[str, object]] = []
        chunks: List[Dict[str, object]] = []

        for path in self.loader.iter_supported_files(self.input_dir):
            logger.info("Ingesting: %s", path.name)
            doc_id = self.identity.build_document_id(path)
            regions = self._build_regions(path)
            doc_regions: List[Dict[str, object]] = []
            doc_chunks: List[Dict[str, object]] = []

            for index, region in enumerate(regions):
                region_id = self.identity.build_region_id(doc_id, index)
                modality = self._modality_for_region(region.type)
                region_meta: Dict[str, Any] = {
                    "source_path": path.name,
                    "region_type": region.type,
                    "modality": modality,
                    "namespace": "tripath-v1",
                    "document_id": doc_id,
                    "region_id": region_id,
                    "page_no": region.page_no,
                    "layout_source": region.layout_source,
                }
                # Carry table-specific structured outputs into region metadata.
                if region.type == "table":
                    if region.table_markdown:
                        region_meta["table_markdown"] = region.table_markdown
                    if region.table_json is not None:
                        region_meta["table_json"] = json.dumps(
                            region.table_json, ensure_ascii=False
                        )
                    # Merge any extra metadata set by TableSerializer.
                    if region.metadata:
                        for k in ("table_html", "table_markdown", "table_json"):
                            if k in region.metadata:
                                region_meta[k] = region.metadata[k]

                doc_regions.append({
                    "id": region_id,
                    "type": region.type,
                    "text": region.text,
                    "start": region.start,
                    "end": region.end,
                    "page_no": region.page_no,
                    "bbox": list(region.bbox) if region.bbox else None,
                    "layout_source": region.layout_source,
                    "table_markdown": region.table_markdown,
                    "table_json": region.table_json,
                    "metadata": region_meta,
                })

                chunk: Dict[str, Any] = {
                    "id": self.identity.build_chunk_id(region_id, len(doc_chunks)),
                    "document_id": doc_id,
                    "region_id": region_id,
                    "modality": modality,
                    "text": region.text,
                    "metadata": region_meta,
                }
                chunks.append(chunk)
                doc_chunks.append(chunk)

            title = self._title_from_regions(regions) or path.stem
            documents.append({
                "id": doc_id,
                "source": path.name,
                "title": title,
                "regions": doc_regions,
                "chunks": doc_chunks,
                "metadata": {
                    "document_id": doc_id,
                    "source_path": path.name,
                    "chunk_count": len(doc_chunks),
                    "namespace": "tripath-v1",
                },
            })

        corpus = {"documents": documents}
        index = {"chunks": chunks}
        audit = {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "notes": [
                "Vision-based layout parsing via Docling deep parser (TableFormer + DocLayNet).",
                "EasyOCR fallback active for scanned / image-only PDF pages.",
                "Table regions serialized to Markdown, HTML, and JSON schema for SQL generation.",
                "Figure captioning via BLIP-2 / CLIP / metadata fallback chain.",
                "FAISS dense indices (all-MiniLM-L6-v2) built per modality.",
                "BM25S sparse index built for text and table modalities.",
                "Loader accepts: PDF, DOCX, PPTX, XLSX, HTML, CSV, plain text.",
            ],
        }

        (self.output_dir / "corpus.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")
        (self.output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        (self.output_dir / "quality_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        self.artifact_writer.write_manifest(documents, chunks)

        # --- Build dense (FAISS) and sparse (BM25S) indices ------------------
        self._build_indices(chunks)

        return {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "output_dir": str(self.output_dir),
        }

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    @trace_execution(level=logging.INFO, name="DocuReasonPipeline._build_indices")
    def _build_indices(self, chunks: List[Dict[str, Any]]) -> None:
        """Build FAISS and BM25S indices from the chunk list.

        Groups chunks by modality, then calls DenseIndexBuilder and
        BM25SIndexBuilder for each group.  All failures are caught and
        logged so a missing optional dependency never crashes the pipeline.
        """
        modality_groups: Dict[str, List[Dict[str, Any]]] = {
            "text": [], "table": [], "vision": []
        }
        for chunk in chunks:
            m = chunk.get("modality", "text")
            if m in modality_groups:
                modality_groups[m].append(chunk)

        for modality, records in modality_groups.items():
            if not records:
                logger.info("Skipping index build for empty modality: %s", modality)
                continue
            try:
                self.dense_index.build(records, modality=modality)
                logger.info(
                    "FAISS index built: modality=%s (%d records)",
                    modality, len(records),
                )
            except Exception as exc:
                logger.warning("FAISS build failed (%s): %s", modality, exc)

            # BM25 only meaningful for text-bearing modalities
            if modality in ("text", "table"):
                try:
                    self.sparse_index.build(records, modality=modality)
                    logger.info(
                        "BM25S index built: modality=%s (%d records)",
                        modality, len(records),
                    )
                except Exception as exc:
                    logger.warning("BM25S build failed (%s): %s", modality, exc)

    # ------------------------------------------------------------------
    # Region construction — replaces the old heuristic _segment_regions()
    # ------------------------------------------------------------------

    @trace_execution(level=logging.INFO, name="DocuReasonPipeline._build_regions")
    def _build_regions(self, path: Path) -> List[Region]:
        """Parse *path* into typed regions using the vision pipeline.

        Strategy
        ---------
        1. Try ``DoclingLayoutParser`` (layout-aware, deep mode).
        2. If it returns nothing, fall back to rule-based text segmentation.
        3. Run ``OCRFallback`` to supplement any empty/scanned pages (PDF only).
        4. Enrich every ``table`` region with ``TableSerializer``.
        5. Caption every ``figure`` region with ``FigureCaptioner``.
        """
        regions: List[Region] = []
        raw_tables: Dict[int, Any] = {}

        # Step 1 — vision-based layout parse.
        if path.suffix.lower() in {".pdf", ".docx", ".pptx", ".xlsx"}:
            try:
                regions, raw_tables = self.layout_parser.parse(path)
                logger.debug(
                    "DoclingLayoutParser: %d region(s) from %s", len(regions), path.name
                )
            except Exception as exc:
                logger.warning("Layout parser failed for %s: %s", path.name, exc)

        # Step 2 — rule-based fallback when Docling produced nothing.
        if not regions:
            logger.info(
                "Falling back to rule-based segmentation for %s", path.name
            )
            payload = self.loader.load(path)
            text = str(payload.get("text", ""))
            regions = self._segment_regions_rule_based(text)

        # Step 3 — OCR supplement for scanned pages (PDF only).
        if path.suffix.lower() == ".pdf":
            try:
                ocr_regions = self.ocr_fallback.run(path, regions)
                if ocr_regions:
                    logger.info(
                        "OCRFallback added %d region(s) from %s",
                        len(ocr_regions),
                        path.name,
                    )
                regions.extend(ocr_regions)
            except Exception as exc:
                logger.warning("OCRFallback error for %s: %s", path.name, exc)

        # Step 4 — table enrichment.
        for idx, region in enumerate(regions):
            if region.type == "table":
                table_item = raw_tables.get(idx)
                if table_item is not None:
                    try:
                        region = self.table_serializer.enrich(region, table_item)
                        regions[idx] = region
                    except Exception as exc:
                        logger.warning(
                            "TableSerializer failed for region %d in %s: %s",
                            idx, path.name, exc,
                        )
                else:
                    # No raw TableItem — try enriching from region.text (GFM fallback).
                    try:
                        from src.tripath.ingestion.table_serializer import _markdown_to_grid
                        grid = _markdown_to_grid(region.text)
                        if len(grid) >= 2:
                            headers = grid[0]
                            rows = grid[1:]
                            region.table_json = {"columns": headers, "rows": rows}
                            region.table_markdown = region.text
                            regions[idx] = region
                    except Exception:
                        pass

        # Step 5 — figure captioning (BLIP-2 → CLIP → metadata fallback).
        try:
            doc_title = next(
                (r.text.strip() for r in regions if r.type == "title" and r.text.strip()),
                "",
            )
            regions = self.figure_captioner.caption_figures(
                path, regions, doc_title=doc_title
            )
        except Exception as exc:
            logger.warning("FigureCaptioner error for %s: %s", path.name, exc)

        return regions

    # ------------------------------------------------------------------
    # Rule-based segmentation (legacy fallback — kept for non-Docling paths)
    # ------------------------------------------------------------------

    def _segment_regions_rule_based(self, text: str) -> List[Region]:
        """Heuristic line-by-line segmentation.  Used only when Docling fails."""
        lines = [line.rstrip() for line in text.splitlines()]
        regions: List[Region] = []
        if not lines:
            return regions

        # First non-empty, non-special line becomes the title.
        title_text = ""
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.lower().startswith(("table:", "figure:")):
                title_text = stripped
                break
        if title_text:
            regions.append(Region(
                type="title", text=title_text,
                start=0, end=len(title_text), layout_source="rule",
            ))

        current_type = "body"
        current_lines: List[str] = []
        current_start = 0

        def flush() -> None:
            nonlocal current_lines, current_type, current_start
            if current_lines:
                block = "\n".join(current_lines).strip()
                if block:
                    regions.append(Region(
                        type=current_type, text=block,
                        start=current_start,
                        end=current_start + len(block),
                        layout_source="rule",
                    ))
                current_lines = []
                current_type = "body"
                current_start = 0

        for idx, line in enumerate(lines[1:], start=1):
            stripped = line.strip()
            if not stripped:
                if current_lines:
                    flush()
                continue
            lower = stripped.lower()
            if lower.startswith("table:"):
                flush()
                current_type = "table"
                current_lines = [stripped]
                current_start = idx
            elif lower.startswith("figure:"):
                flush()
                current_type = "figure"
                current_lines = [stripped]
                current_start = idx
            else:
                if not current_lines:
                    current_start = idx
                current_lines.append(stripped)

        flush()
        return regions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _title_from_regions(regions: List[Region]) -> Optional[str]:
        for r in regions:
            if r.type == "title" and r.text.strip():
                return r.text.strip()
        return None

    @staticmethod
    def _modality_for_region(region_type: str) -> str:
        if region_type == "table":
            return "table"
        if region_type == "figure":
            return "vision"
        return "text"


# Backward compatibility alias
Phase1Pipeline = DocuReasonPipeline


from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from docureason.tripath.config import DocuReasonConfig
from docureason.tripath.evaluation.artifact_quality import ArtifactQualityAuditor
from docureason.tripath.indexing.artifact_writer import ArtifactWriter, build_provenance
from docureason.tripath.indexing.dense_index import DenseIndexBuilder
from docureason.tripath.indexing.sparse_index import BM25SIndexBuilder
from docureason.tripath.ingestion.chunker import SectionAwareChunker
from docureason.tripath.ingestion.docling_layout_parser import DoclingLayoutParser
from docureason.tripath.ingestion.figure_captioner import FigureCaptioner
from docureason.tripath.ingestion.format_loader import FormatAwareLoader
from docureason.tripath.ingestion.identity import IdentityManager
from docureason.tripath.ingestion.ocr_fallback import OCRFallback
from docureason.tripath.ingestion.schema import Chunk, Document, Region
from docureason.tripath.ingestion.table_serializer import TableSerializer
from docureason.tripath.utils import (
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
    4. **Section-aware chunking** (``SectionAwareChunker``) — regions are bound to
       their headings and windowed into overlapping child chunks carrying a
       breadcrumb ancestry header and a link to their parent text.

    On (4): the pipeline previously emitted **one chunk per region**, so parent
    expansion at retrieval time was inert (parent and child were the same string),
    the indexed text carried no breadcrumb, and a long table's markdown blew past
    the embedding model's 256-token limit and was silently truncated.  The chunker
    existed, was tested, and was called by two scripts but never by this pipeline.
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        config: Optional[DocuReasonConfig | str] = None,
        verbose: bool = True,
    ) -> None:
        # Validate minimum required developer inputs
        validation_info = DocuReasonConfig.validate_inputs(input_dir, output_dir)
        self.input_dir = Path(validation_info["input_dir"])
        self.output_dir = Path(validation_info["output_dir"])

        # Resolve configuration object
        if isinstance(config, DocuReasonConfig):
            self.config = config
        elif isinstance(config, str):
            self.config = DocuReasonConfig.load_preset(config)
        else:
            self.config = DocuReasonConfig.load_from_yaml()

        self.verbose = verbose
        if self.verbose:
            self.config.print_summary(input_dir=self.input_dir, output_dir=self.output_dir, logger=logger)

        logger.info("Initializing DocuReasonPipeline: input_dir=%s, output_dir=%s", self.input_dir, self.output_dir)

        # The index builders prefer config.output_dir over their positional argument,
        # so the config must be pointed at the requested run directory *before* they are
        # constructed.  Without this, corpus.json / index.json / manifest.json were written
        # to output_dir while the FAISS and BM25S indexes went to the config default
        # ("artifacts/"), so indexes never lived beside the corpus they indexed and
        # concurrent runs silently overwrote each other's index files.
        self.config.output_dir = str(self.output_dir)

        self.identity = IdentityManager()
        self.artifact_writer = ArtifactWriter(self.output_dir)
        self.loader = FormatAwareLoader()

        self.layout_parser = DoclingLayoutParser(
            page_batch_size=self.config.ingestion.page_batch_size,
            do_ocr=False,
            enable_table_structure=self.config.ingestion.enable_table_structure,
        )
        log_pipeline_flag(
            "layout_parser_batch_size",
            self.config.ingestion.page_batch_size,
            "Docling single-page memory limit",
            logger,
        )

        self.ocr_fallback = OCRFallback(
            languages=["en"],
            min_chars_threshold=self.config.ingestion.ocr_fallback_char_threshold,
        )
        log_pipeline_flag(
            "ocr_languages",
            ["en"],
            f"EasyOCR fallback (threshold={self.config.ingestion.ocr_fallback_char_threshold} chars, "
            f"enabled={self.config.ingestion.enable_ocr_fallback})",
            logger,
        )

        self.table_serializer = TableSerializer()
        self.chunker = SectionAwareChunker.from_config(self.config)
        log_pipeline_flag(
            "child_chunk_size",
            self.config.ingestion.child_chunk_size,
            f"Section-aware child window in words (overlap={self.config.ingestion.chunk_overlap}, "
            f"parent_cap={self.config.ingestion.parent_region_size})",
            logger,
        )
        self.quality_auditor = ArtifactQualityAuditor()

        self.figure_captioner = FigureCaptioner(use_blip2=True, use_clip=True)
        log_pipeline_flag("use_blip2", True, "BLIP-2 figure captioning enabled", logger)
        log_pipeline_flag("use_clip", True, "CLIP figure embeddings enabled", logger)

        self.dense_index = DenseIndexBuilder(output_dir=self.output_dir, config=self.config)
        self.sparse_index = BM25SIndexBuilder(output_dir=self.output_dir, config=self.config)

    @trace_pipeline_stage("Document Ingestion & Indexing Pipeline")
    def run(self) -> Dict[str, object]:
        document_objects: List[Document] = []

        for path in self.loader.iter_supported_files(self.input_dir):
            logger.info("Ingesting: %s", path.name)
            doc_id = self.identity.build_document_id(path)
            regions = self._build_regions(path)
            self._stamp_region_metadata(path, doc_id, regions)

            document = Document(
                id=doc_id,
                source=path.name,
                title=self._title_from_regions(regions) or path.stem,
                regions=regions,
                metadata={
                    "document_id": doc_id,
                    "source_path": path.name,
                    "namespace": "tripath-v1",
                },
            )
            # Section-aware child chunks with breadcrumb headers and parent links.
            document.chunks = self._enrich_chunks(document, self.chunker.chunk_document(document))
            if document.metadata is not None:
                document.metadata["chunk_count"] = len(document.chunks)
            logger.info(
                "%s → %d region(s), %d chunk(s)",
                path.name,
                len(document.regions),
                len(document.chunks),
            )
            document_objects.append(document)

        documents = [self._document_to_dict(doc) for doc in document_objects]
        chunks = [self._chunk_to_dict(chunk) for doc in document_objects for chunk in doc.chunks]

        corpus = {"documents": documents}
        index = {"chunks": chunks}

        # --- Build dense (FAISS) and sparse (BM25S) indices ------------------
        # Built before the audit is written so index_stats records what actually
        # landed on disk rather than what was requested.
        index_stats = self._build_indices(chunks)

        audit = self._build_audit(document_objects, index_stats)

        (self.output_dir / "corpus.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")
        (self.output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        (self.output_dir / "quality_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        self.artifact_writer.write_manifest(documents, chunks, provenance=audit["provenance"])

        return {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "output_dir": str(self.output_dir),
            "index_stats": index_stats,
            "degradation": audit["degradation"],
        }

    # ------------------------------------------------------------------
    # Artifact assembly
    # ------------------------------------------------------------------

    def _stamp_region_metadata(self, path: Path, doc_id: str, regions: List[Region]) -> None:
        """Attach canonical identity and provenance metadata to each ``Region``.

        Stamping the objects (rather than only the serialized dicts) is what lets
        the chunker and every retriever resolve the same ``region_id`` through
        ``canonical_region_id`` instead of reconstructing it with their own offset.
        """
        for index, region in enumerate(regions):
            merged: Dict[str, Any] = dict(region.metadata or {})
            merged.update(
                {
                    "source_path": path.name,
                    "region_type": region.type,
                    "modality": self._modality_for_region(region.type),
                    "namespace": "tripath-v1",
                    "document_id": doc_id,
                    "region_id": self.identity.build_region_id(doc_id, index),
                    "page_no": region.page_no,
                    "layout_source": region.layout_source,
                }
            )
            if region.type == "table":
                if region.table_markdown:
                    merged["table_markdown"] = region.table_markdown
                if region.table_json is not None:
                    merged["table_json"] = json.dumps(region.table_json, ensure_ascii=False)
            region.metadata = merged

    @staticmethod
    def _enrich_chunks(document: Document, chunks: List[Chunk]) -> List[Chunk]:
        """Carry each chunk's owning-region provenance onto the chunk metadata."""
        region_meta = {
            (region.metadata or {}).get("region_id"): (region.metadata or {}) for region in document.regions
        }
        carried = ("source_path", "document_id", "namespace", "page_no", "layout_source")
        for chunk in chunks:
            meta = dict(chunk.metadata or {})
            parent = region_meta.get(chunk.region_id, {})
            for key in carried:
                meta.setdefault(key, parent.get(key))
            chunk.metadata = meta
        return chunks

    def _document_to_dict(self, document: Document) -> Dict[str, Any]:
        return {
            "id": document.id,
            "source": document.source,
            "title": document.title,
            "regions": [
                {
                    "id": (region.metadata or {}).get("region_id"),
                    "type": region.type,
                    "text": region.text,
                    "start": region.start,
                    "end": region.end,
                    "page_no": region.page_no,
                    "bbox": list(region.bbox) if region.bbox else None,
                    "layout_source": region.layout_source,
                    "table_markdown": region.table_markdown,
                    "table_json": region.table_json,
                    "metadata": region.metadata or {},
                }
                for region in document.regions
            ],
            "chunks": [self._chunk_to_dict(chunk) for chunk in document.chunks],
            "metadata": document.metadata or {},
        }

    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> Dict[str, Any]:
        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "region_id": chunk.region_id,
            "modality": chunk.modality,
            "text": chunk.text,
            "metadata": chunk.metadata or {},
        }

    def _build_audit(
        self,
        document_objects: List[Document],
        index_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble a fully *measured* quality audit.

        Two things changed here.  ``ArtifactQualityAuditor`` was implemented,
        tested, exported from the package — and never called; the pipeline wrote its
        own audit instead.  And that audit's ``notes`` were seven hard-coded strings
        asserting that Docling, TableFormer, EasyOCR, BLIP-2 and FAISS had all run,
        emitted unconditionally regardless of what happened on the machine.  Every
        value below is an observation.
        """
        layout_source_counts: Dict[str, int] = {}
        for document in document_objects:
            for region in document.regions:
                layout_source_counts[region.layout_source] = layout_source_counts.get(region.layout_source, 0) + 1

        structured_tables = sum(
            1
            for document in document_objects
            for region in document.regions
            if region.table_json or region.table_markdown
        )

        audit: Dict[str, Any] = dict(self.quality_auditor.audit_documents(document_objects))
        audit["document_count"] = len(document_objects)
        audit["chunk_count"] = sum(len(doc.chunks) for doc in document_objects)
        audit["capabilities_used"] = {
            "regions_by_layout_source": layout_source_counts,
            "structured_tables_serialized": structured_tables,
            "table_structure_enabled": self.layout_parser.table_structure_enabled,
            "table_structure_requested": self.config.ingestion.enable_table_structure,
            "caption_tier_counts": dict(self.figure_captioner.tier_counts),
            "ocr_fallback_enabled": self.config.ingestion.enable_ocr_fallback,
            "ocr_char_threshold": self.config.ingestion.ocr_fallback_char_threshold,
            "chunking": {
                "child_chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.overlap,
                "parent_max_words": self.chunker.parent_max_words,
                "unit": "words",
            },
            "dense_index_model": self.dense_index.model_name,
            "dense_index_revision": self.dense_index.model_revision or "hub-default",
            "dense_index_type": self.dense_index.index_type,
            "bm25_params": {"k1": self.sparse_index.bm25_k1, "b": self.sparse_index.bm25_b},
            "index_stats": index_stats,
        }
        audit["degradation"] = self._ingestion_degradation(layout_source_counts, index_stats)
        audit["provenance"] = build_provenance({"preset": self.config.preset})
        if audit["degradation"]:
            logger.warning("Ingestion degradation: %s", ", ".join(audit["degradation"]))
        return audit

    def _ingestion_degradation(
        self,
        layout_source_counts: Dict[str, int],
        index_stats: Dict[str, Any],
    ) -> List[str]:
        """Every ingestion-time fallback that fired, as short ``subsystem:reason`` labels."""
        labels: List[str] = []

        total_regions = sum(layout_source_counts.values())
        rule_regions = layout_source_counts.get("rule", 0)
        if total_regions and rule_regions / total_regions > 0.5:
            labels.append(f"layout:rule_fallback={rule_regions}/{total_regions}")
        if self.layout_parser.table_structure_enabled is False:
            labels.append("table_structure:disabled")
        if self.layout_parser.table_structure_enabled is None and total_regions:
            labels.append("layout:docling_never_ran")

        tiers = self.figure_captioner.tier_counts
        if tiers.get("metadata"):
            labels.append(f"captions:metadata_fallback={tiers['metadata']}")

        for modality, stats in (index_stats.get("dense") or {}).items():
            if stats.get("degraded"):
                labels.append(f"dense_index:{modality}:{stats['degraded']}")
        for modality, stats in (index_stats.get("sparse") or {}).items():
            if stats.get("degraded"):
                labels.append(f"sparse_index:{modality}:{stats['degraded']}")

        return labels

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    @trace_execution(level=logging.INFO, name="DocuReasonPipeline._build_indices")
    def _build_indices(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build FAISS and BM25S indices from the chunk list and report what landed.

        Failure policy: a *missing optional dependency* degrades (recorded in the
        returned stats and in ``quality_audit.json``), but a build that fails with
        its dependencies present raises.  The builders used to catch everything and
        write a zero-byte index, which at query time is indistinguishable from "no
        relevant documents" — the worst possible failure shape, and one that would
        pair an index with the wrong corpus once ``search()`` is actually called.
        """
        modality_groups: Dict[str, List[Dict[str, Any]]] = {"text": [], "table": [], "vision": []}
        for chunk in chunks:
            m = chunk.get("modality", "text")
            if m in modality_groups:
                modality_groups[m].append(chunk)

        stats: Dict[str, Any] = {"dense": {}, "sparse": {}}

        for modality, records in modality_groups.items():
            if not records:
                logger.info("Skipping index build for empty modality: %s", modality)
                continue

            self.dense_index.build(records, modality=modality)
            dense_stat = dict(self.dense_index.build_stats.get(modality, {}))
            stats["dense"][modality] = dense_stat
            logger.info(
                "FAISS index: modality=%s records=%d ntotal=%s degraded=%s",
                modality,
                len(records),
                dense_stat.get("ntotal"),
                dense_stat.get("degraded"),
            )

            # BM25 is only meaningful for text-bearing modalities.
            if modality in ("text", "table"):
                self.sparse_index.build(records, modality=modality)
                sparse_stat = dict(self.sparse_index.build_stats.get(modality, {}))
                stats["sparse"][modality] = sparse_stat
                logger.info(
                    "BM25S index: modality=%s records=%d indexed=%s degraded=%s",
                    modality,
                    len(records),
                    sparse_stat.get("indexed"),
                    sparse_stat.get("degraded"),
                )

        return stats

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
                logger.debug("DoclingLayoutParser: %d region(s) from %s", len(regions), path.name)
            except Exception as exc:
                logger.warning("Layout parser failed for %s: %s", path.name, exc)

        # Step 2 — rule-based fallback when Docling produced nothing.
        if not regions:
            logger.info("Falling back to rule-based segmentation for %s", path.name)
            payload = self.loader.load(path)
            text = str(payload.get("text", ""))
            regions = self._segment_regions_rule_based(text)

        # Step 3 — OCR supplement for scanned pages (PDF only, config-gated).
        if path.suffix.lower() == ".pdf" and self.config.ingestion.enable_ocr_fallback:
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
                            idx,
                            path.name,
                            exc,
                        )
                else:
                    # No raw TableItem — try enriching from region.text (GFM fallback).
                    try:
                        from docureason.tripath.ingestion.table_serializer import _markdown_to_grid

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
            regions = self.figure_captioner.caption_figures(path, regions, doc_title=doc_title)
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
            regions.append(
                Region(
                    type="title",
                    text=title_text,
                    start=0,
                    end=len(title_text),
                    layout_source="rule",
                )
            )

        current_type = "body"
        current_lines: List[str] = []
        current_start = 0

        def flush() -> None:
            nonlocal current_lines, current_type, current_start
            if current_lines:
                block = "\n".join(current_lines).strip()
                if block:
                    regions.append(
                        Region(
                            type=current_type,
                            text=block,
                            start=current_start,
                            end=current_start + len(block),
                            layout_source="rule",
                        )
                    )
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

"""docling_layout_parser.py — Vision-based document layout segmentation.

Replaces the string-prefix heuristic ``_segment_regions()`` with Docling's
deep layout pipeline (TableFormer + DocLayNet classifier).  Each ``DocItem``
in the conversion result is mapped to a typed ``Region`` with accurate
``page_no``, ``bbox``, and ``layout_source="docling"`` provenance.

Supported region types produced
--------------------------------
* ``"title"``   — SectionHeaderItem (or the first TextItem when no heading exists)
* ``"body"``    — TextItem, ListItem, InlineGroup
* ``"table"``   — TableItem (raw TableData exposed via item reference)
* ``"figure"``  — PictureItem / FigureItem

Usage
-----
::

    from src.tripath.ingestion.docling_layout_parser import DoclingLayoutParser

    parser = DoclingLayoutParser()
    regions, raw_tables = parser.parse(Path("document.pdf"))
    # ``raw_tables`` is a dict mapping region index → docling TableItem
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.tripath.utils import get_logger, log_pipeline_flag, trace_execution
from .schema import Region

logger = get_logger(__name__)


class DoclingLayoutParser:
    """Parse a document via Docling's deep layout pipeline.

    Parameters
    ----------
    page_batch_size:
        Number of PDF pages processed per Docling batch.  Keep at 2 to stay
        within the memory envelope established by the existing OOM fix.
    do_ocr:
        Whether Docling should attempt its *internal* OCR.  We default to
        ``False`` because ``OCRFallback`` handles scanned pages externally
        with EasyOCR, giving us more control over which pages need OCR.
    """

    def __init__(self, page_batch_size: int = 1, do_ocr: bool = False) -> None:
        self.page_batch_size = page_batch_size
        self.do_ocr = do_ocr
        log_pipeline_flag("docling_page_batch_size", page_batch_size, "Batch size for Docling PDF processing", logger)
        log_pipeline_flag("docling_internal_ocr", do_ocr, "Docling internal OCR flag", logger)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @trace_execution(logger=logger, log_return=True)
    def parse(
        self, path: Path
    ) -> Tuple[List[Region], Dict[int, Any]]:
        """Convert *path* with Docling and return typed regions.

        Returns
        -------
        regions:
            Ordered list of ``Region`` objects extracted from the document.
        raw_tables:
            Mapping of region-list-index → raw Docling ``TableItem`` so that
            ``TableSerializer`` can access the structured cell grid later.
        """
        try:
            result = self._run_docling(path)
            if result is None or not hasattr(result, "document") or result.document is None:
                return [], {}
        except Exception as exc:
            logger.warning("DoclingLayoutParser failed for %s: %s", path.name, exc)
            return [], {}

        regions: List[Region] = []
        raw_tables: Dict[int, Any] = {}

        try:
            items = list(result.document.iterate_items())
        except Exception as exc:
            logger.warning("Could not iterate Docling items for %s: %s", path.name, exc)
            return [], {}

        seen_title = False

        for item, _level in items:
            region = self._item_to_region(item, seen_title)
            if region is None:
                continue
            if region.type == "title":
                seen_title = True
            # Store raw TableItem so TableSerializer can process it.
            if region.type == "table":
                raw_tables[len(regions)] = item
            regions.append(region)

        return regions, raw_tables

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_docling(self, path: Path) -> Any:
        """Run Docling with deep layout options and return the ConversionResult."""
        import gc
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.do_ocr
        pipeline_options.do_table_structure = True   # Enable TableFormer

        # Set single-page batching and thread limits for memory safety
        for field, value in [
            ("generate_page_images", False),
            ("images_scale", 1.0),
            ("page_batch_size", self.page_batch_size),
            ("layout_batch_size", self.page_batch_size),
            ("table_batch_size", self.page_batch_size),
            ("ocr_batch_size", self.page_batch_size),
            ("num_threads", 1),
            ("thread_count", 1),
        ]:
            if hasattr(pipeline_options, field):
                setattr(pipeline_options, field, value)

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        try:
            res = converter.convert(str(path))
            gc.collect()
            return res
        except (BaseException, Exception, RuntimeError, MemoryError) as exc:
            logger.warning("Docling PDF conversion exception for %s: %s", path.name, exc)
            return None

    def _item_to_region(self, item: Any, seen_title: bool) -> Optional[Region]:
        """Map a single Docling DocItem to a ``Region``, or ``None`` to skip."""
        # Lazy imports — docling types are only resolved when the parser runs.
        try:
            from docling.datamodel.document import (
                SectionHeaderItem,
                TextItem,
                TableItem,
                PictureItem,
                ListItem,
            )
        except ImportError:
            # Older docling builds use different class paths — fall back gracefully.
            SectionHeaderItem = None
            TextItem = None
            TableItem = None
            PictureItem = None
            ListItem = None

        item_type = type(item)
        item_type_name = item_type.__name__

        region_type = self._classify(item_type_name, seen_title)
        if region_type is None:
            return None

        text = self._extract_text(item)
        if not text.strip() and region_type != "table":
            return None

        page_no, bbox = self._extract_geometry(item)

        return Region(
            type=region_type,
            text=text,
            page_no=page_no,
            bbox=bbox,
            layout_source="docling",
        )

    @staticmethod
    def _classify(type_name: str, seen_title: bool) -> Optional[str]:
        """Return a tripath region type string from a Docling item class name."""
        if type_name in ("SectionHeaderItem",):
            return "title"
        if type_name in ("TextItem", "ListItem", "InlineGroup"):
            return "body"
        if type_name == "TableItem":
            return "table"
        if type_name in ("PictureItem", "FigureItem"):
            return "figure"
        # Skip structural items (PageItem, GroupItem, etc.)
        return None

    @staticmethod
    def _extract_text(item: Any) -> str:
        """Pull the best available text representation from an item."""
        # TableItem: use export_to_markdown for a concise text fallback.
        if hasattr(item, "export_to_markdown"):
            try:
                md = item.export_to_markdown()
                if md:
                    return md
            except Exception:
                pass
        # Standard text property.
        if hasattr(item, "text") and item.text:
            return str(item.text)
        # Some items expose content as a list of inline elements.
        if hasattr(item, "orig"):
            return str(item.orig)
        return ""

    @staticmethod
    def _extract_geometry(
        item: Any,
    ) -> Tuple[int, Optional[Tuple[float, float, float, float]]]:
        """Return (page_no, bbox) from the item's provenance, if available."""
        page_no = -1
        bbox: Optional[Tuple[float, float, float, float]] = None
        try:
            prov = getattr(item, "prov", None)
            if prov:
                # prov is a list; take the first provenance entry.
                entry = prov[0] if isinstance(prov, list) else prov
                page_no = int(getattr(entry, "page_no", -1))
                bb = getattr(entry, "bbox", None)
                if bb is not None:
                    bbox = (
                        float(bb.l),
                        float(bb.t),
                        float(bb.r),
                        float(bb.b),
                    )
        except Exception:
            pass
        return page_no, bbox

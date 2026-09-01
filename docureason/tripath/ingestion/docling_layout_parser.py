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

    from docureason.tripath.ingestion.docling_layout_parser import DoclingLayoutParser

    parser = DoclingLayoutParser()
    regions, raw_tables = parser.parse(Path("document.pdf"))
    # ``raw_tables`` is a dict mapping region index → docling TableItem
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from docureason.tripath.utils import get_logger, log_pipeline_flag, trace_execution

from .schema import Region

logger = get_logger(__name__)


@contextmanager
def cpu_thread_guard(enabled: bool) -> Iterator[None]:
    """Temporarily pin PyTorch to a single thread, then restore the previous value.

    Docling's CPU path needs single-threaded execution to avoid OpenMP/MKL
    access violations in Conv2D on Windows.  This used to be applied as an
    unscoped side effect of :meth:`DoclingLayoutParser._run_docling` and never
    restored, so *every* subsequent embedding batch and cross-encoder rerank in
    the same process also ran single-threaded — silently undoing any
    performance tuning done elsewhere (audit P1-14).

    ``enabled=False`` is a no-op, so callers on CUDA pay nothing.
    """
    if not enabled:
        yield
        return

    try:
        import torch
    except Exception:  # torch absent — nothing to throttle
        yield
        return

    prev_threads = torch.get_num_threads()
    prev_interop: int | None = None
    try:
        prev_interop = torch.get_num_interop_threads()
    except Exception:
        prev_interop = None

    torch.set_num_threads(1)
    if prev_interop is not None and hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Interop pool already started; it cannot be resized mid-process.
            prev_interop = None

    try:
        yield
    finally:
        torch.set_num_threads(prev_threads)
        if prev_interop is not None:
            try:
                torch.set_num_interop_threads(prev_interop)
            except RuntimeError:
                pass


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
    enable_table_structure:
        Tri-state control over TableFormer table-structure recognition.
        ``True`` forces it on, ``False`` forces it off, and ``None`` (the
        default) auto-resolves to "on when CUDA is available".  Previously this
        was an unconditional CUDA check plus two undocumented environment
        variables, so CPU users silently lost span handling, header hierarchy,
        numeric normalization and the unit multiplier while the README and the
        quality audit still claimed TableFormer had run (audit P0-6).

        The resolved value is logged through ``log_pipeline_flag`` and exposed as
        ``self.table_structure_enabled`` after the first parse so the pipeline can
        record what actually ran.
    """

    def __init__(
        self,
        page_batch_size: int = 1,
        do_ocr: bool = False,
        enable_table_structure: Optional[bool] = None,
    ) -> None:
        self.page_batch_size = page_batch_size
        self.do_ocr = do_ocr
        self.enable_table_structure = enable_table_structure
        #: Resolved value from the most recent parse; ``None`` until one has run.
        self.table_structure_enabled: Optional[bool] = None
        log_pipeline_flag("docling_page_batch_size", page_batch_size, "Batch size for Docling PDF processing", logger)
        log_pipeline_flag("docling_internal_ocr", do_ocr, "Docling internal OCR flag", logger)
        log_pipeline_flag(
            "docling_table_structure_request",
            "auto" if enable_table_structure is None else enable_table_structure,
            "TableFormer request (auto = on when CUDA present)",
            logger,
        )

    def _resolve_table_structure(self, has_cuda: bool) -> bool:
        """Resolve the tri-state TableFormer setting into a concrete boolean.

        Precedence: explicit constructor value → ``DOCUREASON_FAST_PARSE=1``
        (force off) → ``DOCUREASON_ENABLE_CPU_TABLEFORMER=1`` (force on) →
        CUDA availability.  The environment variables are retained for
        backward compatibility but are no longer the only way to opt in.
        """
        if self.enable_table_structure is not None:
            return bool(self.enable_table_structure)
        if os.getenv("DOCUREASON_FAST_PARSE") == "1":
            return False
        if os.getenv("DOCUREASON_ENABLE_CPU_TABLEFORMER") == "1":
            return True
        return has_cuda

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @trace_execution(logger=logger, log_return=True)
    def parse(self, path: Path) -> Tuple[List[Region], Dict[int, Any]]:
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
        if os.getenv("DOCUREASON_DISABLE_DOCLING") == "1":
            return None

        import gc

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.do_ocr

        # Check CUDA availability to decide TableFormer and thread policy.  The
        # single-threading itself is applied by cpu_thread_guard() around the
        # convert() call only — never process-wide (audit P1-14).
        has_cuda = False
        try:
            import torch

            has_cuda = torch.cuda.is_available()
        except Exception:
            pass

        enable_table_structure = self._resolve_table_structure(has_cuda)
        pipeline_options.do_table_structure = enable_table_structure
        self.table_structure_enabled = enable_table_structure
        log_pipeline_flag(
            "table_structure_enabled",
            enable_table_structure,
            f"TableFormer structure recognition (cuda={has_cuda}, override={self.enable_table_structure})",
            logger,
        )

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

        try:
            with cpu_thread_guard(enabled=not has_cuda):
                converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
                )
                res = converter.convert(str(path))
            gc.collect()
            return res
        except Exception as exc:
            # Narrowed from (BaseException, Exception, RuntimeError, MemoryError):
            # that tuple was redundant (BaseException subsumes the rest) and it
            # swallowed KeyboardInterrupt/SystemExit.  MemoryError is an Exception
            # subclass, so OOM is still handled here.
            logger.warning("Docling PDF conversion exception for %s: %s", path.name, exc)
            return None

    def _item_to_region(self, item: Any, seen_title: bool) -> Optional[Region]:
        """Map a single Docling DocItem to a ``Region``, or ``None`` to skip."""
        try:
            from docling.datamodel.document import (  # noqa: F401
                ListItem,
                PictureItem,
                SectionHeaderItem,
                TableItem,
                TextItem,
            )
        except ImportError:
            # Older docling builds use different class paths — fall back gracefully.
            pass

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

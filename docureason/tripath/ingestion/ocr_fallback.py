"""ocr_fallback.py — EasyOCR-based fallback for scanned PDF pages.

When Docling's layout parser returns no text for one or more pages
(i.e., the page is image-only / scanned), this module renders those pages
to raster images using ``pypdf``'s built-in image extraction *or* a pure
fallback via ``pillow``, then passes them through EasyOCR.

The resulting text is returned as ``Region`` objects with
``layout_source="ocr"`` so callers can distinguish them from
vision-parsed regions.

Design decisions
----------------
* **Lazy EasyOCR init**: The ``easyocr.Reader`` is expensive to load
  (~1–2 s GPU warmup).  We instantiate it once and cache it on the class.
* **Page-emptiness heuristic**: A page is considered "scanned/empty" when
  the total character count of all Docling regions for that page is below
  ``min_chars_threshold`` (default 20).
* **No poppler dependency**: We use ``pypdf`` page images when present,
  falling back to rendering via ``pillow`` if no embedded images exist.
  This avoids the need for a system-level ``poppler`` binary.

Usage
-----
::

    from docureason.tripath.ingestion.ocr_fallback import OCRFallback

    fallback = OCRFallback(languages=["en"])
    extra_regions = fallback.run(pdf_path, regions_from_docling)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from docureason.tripath.utils import get_logger, log_pipeline_flag, trace_execution

from .rasterizer import PageRasterizer, page_index_from_region
from .schema import Region

logger = get_logger(__name__)

# How many characters a page must have to be considered "non-scanned".
_MIN_CHARS_THRESHOLD = 20


class OCRFallback:
    """EasyOCR fallback for scanned / image-only PDF pages.

    Parameters
    ----------
    languages:
        List of ISO 639-1 language codes passed to ``easyocr.Reader``.
        Defaults to ``["en"]``.
    gpu:
        Whether to use GPU acceleration.  ``None`` lets EasyOCR decide.
    min_chars_threshold:
        Pages whose total character count from Docling is below this number
        are treated as scanned and sent through OCR.
    """

    _reader_cache: Optional[Any] = None  # class-level cache to avoid re-init

    def __init__(
        self,
        languages: List[str] | None = None,
        gpu: bool | None = None,
        min_chars_threshold: int = _MIN_CHARS_THRESHOLD,
        image_rendering_dpi: int = 150,
    ) -> None:
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.min_chars_threshold = min_chars_threshold
        #: DPI for the page rasters handed to EasyOCR. OCR accuracy is sensitive to
        #: resolution, and this setting previously reached nothing at all.
        self.image_rendering_dpi = int(image_rendering_dpi)
        log_pipeline_flag(
            "ocr_min_chars_threshold", min_chars_threshold, "Character threshold to detect scanned PDF pages", logger
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @trace_execution(logger=logger, log_return=True)
    def run(self, pdf_path: Path, existing_regions: List[Region]) -> List[Region]:
        """Identify scanned pages in *pdf_path* and OCR them.

        Parameters
        ----------
        pdf_path:
            Path to the PDF file that was already processed by Docling.
        existing_regions:
            Regions already extracted (by ``DoclingLayoutParser``) — used
            to detect which pages are empty.

        Returns
        -------
        List of supplemental ``Region`` objects for pages that needed OCR.
        These should be *appended* to (not replace) the existing regions.
        """
        if pdf_path.suffix.lower() != ".pdf":
            return []

        page_char_counts = self._count_chars_per_page(existing_regions)
        scanned_pages = self._detect_scanned_pages(pdf_path, page_char_counts)

        if not scanned_pages:
            logger.debug("No scanned pages detected in %s", pdf_path.name)
            return []

        logger.info(
            "OCRFallback: %d scanned page(s) detected in %s — running EasyOCR",
            len(scanned_pages),
            pdf_path.name,
        )

        reader = self._get_reader()
        if reader is None:
            logger.warning("EasyOCR unavailable — skipping OCR fallback")
            return []

        ocr_regions: List[Region] = []
        for page_index, page_image_bytes in scanned_pages.items():
            text = self._ocr_image(reader, page_image_bytes)
            if text.strip():
                ocr_regions.append(
                    Region(
                        type="body",
                        text=text.strip(),
                        # Emitted 1-based, matching Docling's convention: these regions
                        # are appended to Docling's, and two page-numbering conventions
                        # in one document is how the off-by-one above came about.
                        page_no=page_index + 1,
                        layout_source="ocr",
                    )
                )
                logger.debug("OCRFallback: page index %d -> %d chars", page_index, len(text))

        return ocr_regions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_chars_per_page(regions: List[Region], page_count: int = 0) -> Dict[int, int]:
        """Characters per **0-based page index**, from the regions already extracted.

        Docling reports ``page_no`` 1-based while the page loop below indexes from 0
        (audit P1-2).  Counting under one convention and looking up under the other
        meant page N's character count was compared against page N+1's content, so the
        scanned-page detector was reliably off by one page.
        """
        counts: Dict[int, int] = {}
        for region in regions:
            if region.page_no is None or region.page_no < 0:
                continue
            index = page_index_from_region(region.page_no, page_count or region.page_no + 1)
            counts[index] = counts.get(index, 0) + len(region.text)
        return counts

    def _detect_scanned_pages(
        self,
        pdf_path: Path,
        page_char_counts: Dict[int, int],
    ) -> Dict[int, bytes]:
        """Map 0-based page index -> a PNG raster of the whole page, for scanned pages.

        The page is **rendered**, not scavenged.  Taking ``page.images[0]`` returned one
        embedded image per page, so a scan split across two images OCR'd only the first
        half (P2-13) - and when a page had no embedded image at all, the previous code
        handed EasyOCR a blank white canvas, which costs seconds per page to read
        nothing. A rendered page contains whatever is actually on it.
        """
        scanned: Dict[int, bytes] = {}
        rasterizer = PageRasterizer(pdf_path, dpi=self.image_rendering_dpi)
        if not rasterizer.available:
            logger.warning("OCRFallback: cannot rasterize %s — skipping OCR entirely", pdf_path.name)
            return scanned

        try:
            for page_index in range(rasterizer.page_count):
                if page_char_counts.get(page_index, 0) >= self.min_chars_threshold:
                    continue  # Docling already extracted enough text here
                image = rasterizer.render(page_index)
                if image is None:
                    continue
                if PageRasterizer.is_blank(image):
                    logger.debug("OCRFallback: page %d renders blank — nothing to OCR", page_index)
                    continue
                payload = PageRasterizer.to_png_bytes(image)
                if payload:
                    scanned[page_index] = payload
        except Exception as exc:
            logger.warning("Could not inspect pages of %s: %s", pdf_path.name, exc)
        finally:
            rasterizer.close()
        return scanned

    def _get_reader(self) -> Optional[Any]:
        """Return a cached EasyOCR Reader, initialising it on first use."""
        if OCRFallback._reader_cache is not None:
            return OCRFallback._reader_cache
        try:
            import easyocr

            kwargs: Dict[str, Any] = {"lang_list": self.languages}
            if self.gpu is not None:
                kwargs["gpu"] = self.gpu
            OCRFallback._reader_cache = easyocr.Reader(**kwargs)
            return OCRFallback._reader_cache
        except ImportError:
            logger.warning("easyocr is not installed.  Install it with: pip install easyocr")
            return None
        except Exception as exc:
            logger.warning("Failed to initialise EasyOCR reader: %s", exc)
            return None

    @staticmethod
    def _ocr_image(reader: Any, image_bytes: bytes) -> str:
        """Run EasyOCR on raw image bytes and return concatenated text."""
        try:
            try:
                import torch

                with torch.no_grad():
                    results = reader.readtext(image_bytes, detail=0, paragraph=True)
            except ImportError:
                results = reader.readtext(image_bytes, detail=0, paragraph=True)
            return "\n".join(str(r) for r in results)
        except Exception as exc:
            logger.warning("EasyOCR readtext failed: %s", exc)
            return ""

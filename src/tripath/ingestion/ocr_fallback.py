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

    from src.tripath.ingestion.ocr_fallback import OCRFallback

    fallback = OCRFallback(languages=["en"])
    extra_regions = fallback.run(pdf_path, regions_from_docling)
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tripath.utils import get_logger, log_pipeline_flag, trace_execution

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
    ) -> None:
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.min_chars_threshold = min_chars_threshold
        log_pipeline_flag("ocr_min_chars_threshold", min_chars_threshold, "Character threshold to detect scanned PDF pages", logger)

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
        for page_no, page_image_bytes in scanned_pages.items():
            text = self._ocr_image(reader, page_image_bytes)
            if text.strip():
                ocr_regions.append(
                    Region(
                        type="body",
                        text=text.strip(),
                        page_no=page_no,
                        layout_source="ocr",
                    )
                )
                logger.debug(
                    "OCRFallback: page %d → %d chars", page_no, len(text)
                )

        return ocr_regions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_chars_per_page(regions: List[Region]) -> Dict[int, int]:
        """Sum character counts per page_no from existing regions."""
        counts: Dict[int, int] = {}
        for region in regions:
            p = region.page_no
            if p < 0:
                continue
            counts[p] = counts.get(p, 0) + len(region.text)
        return counts

    def _detect_scanned_pages(
        self,
        pdf_path: Path,
        page_char_counts: Dict[int, int],
    ) -> Dict[int, bytes]:
        """Return a mapping of page_no → raw image bytes for scanned pages."""
        scanned: Dict[int, bytes] = {}
        try:
            import pypdf

            reader = pypdf.PdfReader(str(pdf_path))
            for page_no, page in enumerate(reader.pages):
                if page_char_counts.get(page_no, 0) >= self.min_chars_threshold:
                    continue  # Docling already extracted enough text here
                # Try to grab the first embedded raster image from the page.
                image_bytes = self._extract_page_image(page, page_no)
                if image_bytes:
                    scanned[page_no] = image_bytes
        except Exception as exc:
            logger.warning("Could not inspect pages of %s: %s", pdf_path.name, exc)
        return scanned

    @staticmethod
    def _extract_page_image(page: Any, page_no: int) -> Optional[bytes]:
        """Extract the first raster image embedded in a pypdf page object.

        Falls back to a blank-white PIL image rendered at a fixed DPI when no
        embedded images are found (so EasyOCR at least gets something).
        """
        try:
            images = list(page.images)
            if images:
                return images[0].data  # raw bytes (JPEG / PNG etc.)
        except Exception:
            pass

        # Fallback: render page as a plain white PIL image.
        try:
            from PIL import Image

            width = int(float(page.mediabox.width))
            height = int(float(page.mediabox.height))
            img = Image.new("RGB", (max(width, 100), max(height, 100)), color=(255, 255, 255))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:
            logger.debug("Could not render fallback image for page %d: %s", page_no, exc)
            return None

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
            logger.warning(
                "easyocr is not installed.  Install it with: pip install easyocr"
            )
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

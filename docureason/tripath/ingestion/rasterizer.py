"""rasterizer.py — render PDF pages, so figure and OCR images are real.

Three defects shared one root cause: nothing in the pipeline could rasterize a page.

* **P1-3** - the figure captioner returned ``page.images[0]`` for *every* figure on a
  page, so three charts on one page were captioned from whichever image happened to be
  first, and two of the three captions described the wrong chart.
* **P1-4** - when there was no embedded image, it built a **blank white canvas**,
  cropped it to the bbox and captioned that.  BLIP-2 dutifully described an empty
  rectangle and the result was indexed as a figure.
* **P2-13** - the OCR fallback also took only the first embedded image per page, so a
  scanned page whose text sat in the second image OCR'd to nothing.

``pypdfium2`` (already installed - Docling depends on it) renders a page to a bitmap,
which is what all three needed.  ``IngestionConfig.image_rendering_dpi`` finally has a
consumer: it was a documented, banner-printed setting with no effect anywhere (§6).

Coordinates
-----------
Docling reports a bbox as ``(left, top, right, bottom)`` in PDF points, in a
**bottom-left** origin, so ``top > bottom`` numerically.  PIL crops in pixels from a
**top-left** origin.  :meth:`PageRasterizer.crop` does that conversion in one place;
it was previously open-coded in the captioner with the coordinates named ``(x0, y0,
x1, y1)``, which is how the flip came to be applied to the wrong pair.

Page indexing
-------------
Docling provenance ``page_no`` is **1-based**; ``pypdfium2`` and ``pypdf`` index pages
from 0 (audit P1-2).  :func:`page_index_from_region` performs that single conversion,
clamped to the document, rather than each caller guessing.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple

from docureason.tripath.utils import get_logger

logger = get_logger(__name__)

#: PDF user space is defined at 72 points per inch.
POINTS_PER_INCH = 72.0

#: A bbox narrower or shorter than this (in PDF points, before padding) cannot contain
#: a figure. Checked unpadded: with padding included, a 0.2pt-wide box would clear a
#: pixel-based threshold and yield an 8px sliver that is worse than the whole page.
MIN_CROP_POINTS = 8.0

#: A crop whose pixels vary less than this is treated as blank. Chosen well below the
#: variation of any real chart (axes and text alone exceed it) and above JPEG noise on
#: a genuinely empty region.
BLANK_STD_THRESHOLD = 3.0


def page_index_from_region(page_no: Optional[int], page_count: int) -> int:
    """Convert a 1-based Docling ``page_no`` to a 0-based page index, clamped.

    ``page_no`` of -1 (Docling gave no provenance) means "unknown", which resolves to
    the first page - the same thing the callers did before, but stated once.
    """
    if page_count <= 0:
        return 0
    if page_no is None or page_no < 0:
        return 0
    index = page_no - 1 if page_no >= 1 else 0
    return max(0, min(index, page_count - 1))


class PageRasterizer:
    """Render and crop pages of one PDF at a fixed DPI, caching each page.

    Caching matters: a filing page carrying six figures would otherwise be rasterized
    six times, and rasterization is the expensive part of captioning.
    """

    def __init__(self, pdf_path: Any, dpi: int = 150) -> None:
        self.pdf_path = pdf_path
        self.dpi = max(36, int(dpi))
        self.scale = self.dpi / POINTS_PER_INCH
        self._document: Any = None
        self._pages: Dict[int, Any] = {}
        self._page_sizes: Dict[int, Tuple[float, float]] = {}
        self._unavailable = False

    # -- lifecycle ----------------------------------------------------------

    def _open(self) -> Any:
        if self._document is not None or self._unavailable:
            return self._document
        try:
            import pypdfium2

            self._document = pypdfium2.PdfDocument(str(self.pdf_path))
        except Exception as exc:
            logger.warning("PageRasterizer: cannot open %s (%s) — rasterization unavailable", self.pdf_path, exc)
            self._unavailable = True
            self._document = None
        return self._document

    @property
    def available(self) -> bool:
        return self._open() is not None

    @property
    def page_count(self) -> int:
        document = self._open()
        try:
            return len(document) if document is not None else 0
        except Exception:
            return 0

    def close(self) -> None:
        self._pages.clear()
        if self._document is not None:
            try:
                self._document.close()
            except Exception:
                pass
            self._document = None

    # -- rendering ----------------------------------------------------------

    def render(self, page_index: int) -> Optional[Any]:
        """The rendered page as a PIL image, or None when rasterization is unavailable."""
        if page_index in self._pages:
            return self._pages[page_index]

        document = self._open()
        if document is None:
            return None
        try:
            page = document[page_index]
            width_pt, height_pt = page.get_size()
            self._page_sizes[page_index] = (float(width_pt), float(height_pt))
            image = page.render(scale=self.scale).to_pil().convert("RGB")
        except Exception as exc:
            logger.warning("PageRasterizer: page %d of %s failed to render: %s", page_index, self.pdf_path, exc)
            return None

        self._pages[page_index] = image
        return image

    def page_size(self, page_index: int) -> Optional[Tuple[float, float]]:
        """Page size in PDF points, populated as a side effect of rendering."""
        if page_index not in self._page_sizes:
            self.render(page_index)
        return self._page_sizes.get(page_index)

    def crop(
        self,
        page_index: int,
        bbox: Optional[Tuple[float, float, float, float]],
        pad_points: float = 2.0,
    ) -> Optional[Any]:
        """Crop the rendered page to *bbox*, given as Docling ``(l, t, r, b)`` points.

        A missing or degenerate bbox yields the whole page rather than nothing: a
        full-page image is a worse caption input than a tight crop, but it is a real
        image of real content, which a blank canvas was not.
        """
        image = self.render(page_index)
        if image is None:
            return None
        size = self._page_sizes.get(page_index)
        if not bbox or size is None:
            return image

        page_height = size[1]
        left, top, right, bottom = (float(v) for v in bbox)
        # Bottom-left origin: `top` is numerically the larger y. Normalise first so a
        # producer that emits them the other way round still crops the same box.
        y_high, y_low = max(top, bottom), min(top, bottom)
        x_left, x_right = min(left, right), max(left, right)

        if (x_right - x_left) < MIN_CROP_POINTS or (y_high - y_low) < MIN_CROP_POINTS:
            logger.debug("PageRasterizer: bbox %s on page %d is degenerate - using the full page", bbox, page_index)
            return image

        px0 = int(max(0.0, (x_left - pad_points)) * self.scale)
        px1 = int(min(size[0], (x_right + pad_points)) * self.scale)
        py0 = int(max(0.0, (page_height - y_high - pad_points)) * self.scale)
        py1 = int(min(page_height, (page_height - y_low + pad_points)) * self.scale)

        if px1 - px0 < 4 or py1 - py0 < 4:
            return image
        return image.crop((px0, py0, px1, py1))

    # -- inspection ---------------------------------------------------------

    @staticmethod
    def is_blank(image: Any, std_threshold: float = BLANK_STD_THRESHOLD) -> bool:
        """True when *image* carries no visual content worth captioning.

        Uses the standard deviation of the greyscale histogram: a blank canvas has
        zero, and anything with axes, gridlines or text is far above the threshold.
        This is the check whose absence let empty rectangles be captioned and indexed
        as figures (P1-4).
        """
        if image is None:
            return True
        try:
            from PIL import ImageStat

            stat = ImageStat.Stat(image.convert("L"))
            return float(stat.stddev[0]) < std_threshold
        except Exception:
            # Cannot tell: treat as non-blank rather than silently discarding a figure.
            return False

    @staticmethod
    def to_png_bytes(image: Any) -> Optional[bytes]:
        if image is None:
            return None
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as exc:
            logger.debug("PageRasterizer: PNG encode failed: %s", exc)
            return None


__all__ = [
    "BLANK_STD_THRESHOLD",
    "MIN_CROP_POINTS",
    "POINTS_PER_INCH",
    "PageRasterizer",
    "page_index_from_region",
]

"""test_rasterizer.py — real page rasters, and the blank check that was missing.

Three audit findings shared one cause: nothing could rasterize a page, so the captioner
reused one embedded image for every figure (P1-3), captioned blank canvases (P1-4), and
the OCR fallback read only the first image per page (P2-13).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docureason.tripath.ingestion.rasterizer import (
    POINTS_PER_INCH,
    PageRasterizer,
    page_index_from_region,
)

pytestmark = pytest.mark.unit

SAMPLE_PDF = Path(__file__).resolve().parents[2] / "samples" / "9781513563602-mod01.pdf"


@pytest.fixture(scope="module")
def rasterizer():
    if not SAMPLE_PDF.exists():
        pytest.skip("sample PDF not present")
    instance = PageRasterizer(SAMPLE_PDF, dpi=150)
    if not instance.available:
        pytest.skip("pypdfium2 unavailable")
    yield instance
    instance.close()


# ---------------------------------------------------------------------------
# Page indexing (audit P1-2)
# ---------------------------------------------------------------------------


def test_docling_page_numbers_are_converted_to_zero_based_indices() -> None:
    """Docling counts pages from 1; pypdfium2 and pypdf from 0."""
    assert page_index_from_region(1, 9) == 0
    assert page_index_from_region(4, 9) == 3


def test_an_unknown_page_resolves_to_the_first() -> None:
    assert page_index_from_region(-1, 9) == 0
    assert page_index_from_region(None, 9) == 0


def test_an_out_of_range_page_is_clamped_not_wrapped() -> None:
    assert page_index_from_region(99, 9) == 8


def test_an_empty_document_yields_index_zero() -> None:
    assert page_index_from_region(3, 0) == 0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_dpi_determines_the_raster_size(rasterizer) -> None:
    """The first consumer of image_rendering_dpi, which previously reached no code."""
    image = rasterizer.render(0)
    width_pt, height_pt = rasterizer.page_size(0)

    assert image is not None
    assert image.size[0] == pytest.approx(width_pt * 150 / POINTS_PER_INCH, abs=2)
    assert image.size[1] == pytest.approx(height_pt * 150 / POINTS_PER_INCH, abs=2)


def test_a_lower_dpi_produces_a_smaller_raster() -> None:
    if not SAMPLE_PDF.exists():
        pytest.skip("sample PDF not present")
    low = PageRasterizer(SAMPLE_PDF, dpi=72)
    high = PageRasterizer(SAMPLE_PDF, dpi=200)
    try:
        if not (low.available and high.available):
            pytest.skip("pypdfium2 unavailable")
        assert low.render(0).size[0] < high.render(0).size[0]
    finally:
        low.close()
        high.close()


def test_pages_are_rendered_once_and_cached(rasterizer) -> None:
    """A page with six figures must not be rasterized six times."""
    assert rasterizer.render(0) is rasterizer.render(0)


def test_an_unreadable_document_reports_unavailable_rather_than_raising(tmp_path: Path) -> None:
    broken = tmp_path / "not-a-pdf.pdf"
    broken.write_text("this is not a PDF", encoding="utf-8")
    instance = PageRasterizer(broken)
    assert instance.available is False
    assert instance.render(0) is None
    assert instance.page_count == 0


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def test_a_bbox_crop_is_smaller_than_the_page(rasterizer) -> None:
    page = rasterizer.render(0)
    crop = rasterizer.crop(0, (100.0, 700.0, 400.0, 600.0))
    assert crop is not None
    assert crop.size[0] < page.size[0] and crop.size[1] < page.size[1]


def test_bbox_top_and_bottom_may_arrive_either_way_round(rasterizer) -> None:
    """Docling emits (l, t, r, b) bottom-origin, so t > b; normalising avoids a flip."""
    forward = rasterizer.crop(0, (100.0, 700.0, 400.0, 600.0))
    reversed_pair = rasterizer.crop(0, (100.0, 600.0, 400.0, 700.0))
    assert forward.size == reversed_pair.size


def test_a_missing_bbox_yields_the_whole_page(rasterizer) -> None:
    """A full page is a worse caption input than a crop, but it is real content."""
    assert rasterizer.crop(0, None).size == rasterizer.render(0).size


def test_a_degenerate_bbox_yields_the_whole_page(rasterizer) -> None:
    assert rasterizer.crop(0, (100.0, 500.0, 100.2, 500.1)).size == rasterizer.render(0).size


# ---------------------------------------------------------------------------
# Blank detection (audit P1-4)
# ---------------------------------------------------------------------------


def test_a_uniform_canvas_is_blank() -> None:
    from PIL import Image

    assert PageRasterizer.is_blank(Image.new("RGB", (200, 120), (255, 255, 255))) is True
    assert PageRasterizer.is_blank(Image.new("RGB", (200, 120), (0, 0, 0))) is True


def test_an_image_with_content_is_not_blank() -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (200, 120), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 120, 90), outline=(0, 0, 0), width=3)
    draw.line((10, 100, 190, 20), fill=(0, 0, 0), width=2)
    assert PageRasterizer.is_blank(image) is False


def test_none_counts_as_blank() -> None:
    assert PageRasterizer.is_blank(None) is True


def test_a_rendered_text_page_is_not_blank(rasterizer) -> None:
    assert PageRasterizer.is_blank(rasterizer.render(0)) is False


def test_png_encoding_round_trips(rasterizer) -> None:
    payload = PageRasterizer.to_png_bytes(rasterizer.crop(0, (100.0, 700.0, 400.0, 600.0)))
    assert payload and payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert PageRasterizer.to_png_bytes(None) is None

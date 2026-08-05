"""figure_captioner.py — Three-tier figure caption generator.

Turns ``figure`` regions (which have a ``bbox`` and ``page_no`` but no
meaningful ``text``) into semantically useful captions for downstream
embedding and retrieval.

Tier priority
-------------
1. **BLIP-2** (``Salesforce/blip2-opt-2.7b``) — rich free-form VLM caption.
   Requires ``transformers`` + ``torch`` to be installed.  GPU-optional.
2. **CLIP zero-shot** — classifies the figure into one of a fixed label set
   (bar chart, line chart, pie chart, scatter plot, diagram, table,
   photograph, equation).  Requires ``transformers`` + ``torch``.
3. **Metadata fallback** — constructs a deterministic caption from
   ``page_no``, ``bbox``, and the document title.  Zero extra dependencies.

Usage
-----
::

    from src.tripath.ingestion.figure_captioner import FigureCaptioner

    captioner = FigureCaptioner()
    regions = captioner.caption_figures(pdf_path, regions, doc_title="Annual Report")
    # Every figure region now has non-empty region.text and
    # region.metadata["caption"] set.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, List, Optional

from src.tripath.utils import get_logger, log_pipeline_flag, trace_execution
from .schema import Region

logger = get_logger(__name__)

# Fixed CLIP label set for zero-shot chart classification.
_CHART_LABELS = [
    "bar chart",
    "line chart",
    "pie chart",
    "scatter plot",
    "diagram",
    "table image",
    "photograph",
    "equation",
]


class FigureCaptioner:
    """Generate captions for figure regions extracted from documents.

    Parameters
    ----------
    use_blip2:
        Attempt BLIP-2 captioning when ``transformers`` is available.
        Set to ``False`` to skip directly to CLIP or metadata fallback.
    use_clip:
        Attempt CLIP chart classification when ``transformers`` is available.
    """

    def __init__(
        self,
        use_blip2: bool = True,
        use_clip: bool = True,
    ) -> None:
        self.use_blip2 = use_blip2
        self.use_clip = use_clip
        log_pipeline_flag("figure_captioner_blip2", use_blip2, "BLIP-2 figure captioner flag", logger)
        log_pipeline_flag("figure_captioner_clip", use_clip, "CLIP chart classifier flag", logger)

        self._blip2_model: Any = None
        self._blip2_proc: Any = None
        self._clip_model: Any = None
        self._clip_proc: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @trace_execution(logger=logger, log_return=True)
    def caption_figures(
        self,
        path: Path,
        regions: List[Region],
        doc_title: str = "",
    ) -> List[Region]:
        """Populate ``text`` and ``metadata["caption"]`` for every figure region.

        Non-figure regions are returned unmodified.  The original list is
        mutated in-place *and* returned for convenience.

        Parameters
        ----------
        source_path:
            Path to the source document (used to extract page images for PDF).
        regions:
            Mixed list of ``Region`` objects.  Only ``type == "figure"``
            entries are processed.
        doc_title:
            Document title — used for the metadata fallback caption.
        """
        figure_indices = [i for i, r in enumerate(regions) if r.type == "figure"]
        if not figure_indices:
            return regions

        for idx in figure_indices:
            region = regions[idx]
            existing_text = (region.text or "").strip()
            # If region already has descriptive text, keep it as the primary caption.
            if existing_text and len(existing_text) > 8 and existing_text.lower() not in ("figure", "image", "picture"):
                caption = existing_text
            else:
                generated = self._caption_region(region, source_path, doc_title)
                caption = f"{existing_text} - {generated}".strip(" -") if existing_text else generated

            region.text = caption
            if region.metadata is None:
                region.metadata = {}
            region.metadata["caption"] = caption
            regions[idx] = region
            logger.debug(
                "FigureCaptioner: region %d → %r (page %d)",
                idx, caption[:60], region.page_no,
            )

        return regions

    # ------------------------------------------------------------------
    # Caption generation tiers
    # ------------------------------------------------------------------

    def _caption_region(
        self, region: Region, source_path: Path, doc_title: str
    ) -> str:
        """Try each tier in order, returning the first successful caption."""
        image_bytes = self._extract_region_image(region, source_path)

        if image_bytes:
            if self.use_blip2:
                caption = self._blip2_caption(image_bytes)
                if caption:
                    return caption
            if self.use_clip:
                label = self._clip_label(image_bytes)
                if label:
                    return self._label_to_caption(label, region, doc_title)

        return self._metadata_caption(region, doc_title)

    # ------------------------------------------------------------------
    # Image extraction
    # ------------------------------------------------------------------

    def _extract_region_image(
        self, region: Region, source_path: Path
    ) -> Optional[bytes]:
        """Extract image bytes for the region's page / bbox.

        For PDFs: tries embedded images on the page first, then renders a
        blank PIL image as a last resort (so CLIP/BLIP2 at least get *something*).
        For non-PDF formats: returns ``None`` (no image extraction possible).
        """
        if source_path.suffix.lower() != ".pdf":
            return None

        try:
            import pypdf
            from PIL import Image

            reader = pypdf.PdfReader(str(source_path))
            page_no = region.page_no if region.page_no >= 0 else 0
            if page_no >= len(reader.pages):
                page_no = len(reader.pages) - 1
            page = reader.pages[page_no]

            # Try embedded images first.
            try:
                images = list(page.images)
                if images:
                    return images[0].data
            except Exception:
                pass

            # Render a white-canvas crop at the bbox position if available.
            try:
                mb = page.mediabox
                page_w = float(mb.width)
                page_h = float(mb.height)
                scale = 1.5  # ~108 DPI
                pw = max(1, int(page_w * scale))
                ph = max(1, int(page_h * scale))

                if region.bbox:
                    x0, y0, x1, y1 = region.bbox
                    # PDF coords: y=0 at bottom; PIL: y=0 at top
                    cx0 = int(x0 * scale)
                    cy0 = int((page_h - y1) * scale)
                    cx1 = int(x1 * scale)
                    cy1 = int((page_h - y0) * scale)
                    cx0, cx1 = max(0, cx0), min(pw, cx1)
                    cy0, cy1 = max(0, cy0), min(ph, cy1)
                else:
                    cx0, cy0, cx1, cy1 = 0, 0, pw, ph

                canvas = Image.new("RGB", (pw, ph), (255, 255, 255))
                region_img = canvas.crop((cx0, cy0, cx1, cy1))
                buf = io.BytesIO()
                region_img.save(buf, format="PNG")
                return buf.getvalue()
            except Exception as exc:
                logger.debug("Canvas crop failed: %s", exc)
        except Exception as exc:
            logger.debug("Image extraction failed for %s: %s", source_path.name, exc)
        return None

    # ------------------------------------------------------------------
    # Tier 1 — BLIP-2
    # ------------------------------------------------------------------

    def _blip2_caption(self, image_bytes: bytes) -> Optional[str]:
        """Generate a free-form caption using BLIP-2."""
        try:
            torch, processor, model = self._load_blip2()
            if processor is None or model is None:
                return None
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            device = self._resolve_device()
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=60)
            caption = processor.decode(out[0], skip_special_tokens=True).strip()
            return caption if caption else None
        except Exception as exc:
            logger.debug("BLIP-2 captioning failed: %s", exc)
            return None

    def _load_blip2(self):
        """Lazy-load BLIP-2 model + processor (cached on instance)."""
        if self._blip2_processor is not None:
            return self._torch, self._blip2_processor, self._blip2_model
        try:
            import torch
            from transformers import Blip2Processor, Blip2ForConditionalGeneration

            self._torch = torch
            device = self._resolve_device()
            dtype = torch.float16 if device == "cuda" else torch.float32
            self._blip2_processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b"
            )
            self._blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", torch_dtype=dtype
            ).to(device)
            self._blip2_model.eval()
            logger.info("BLIP-2 loaded on %s", device)
        except Exception as exc:
            logger.debug("BLIP-2 unavailable: %s", exc)
            self._blip2_processor = None
            self._blip2_model = None
        return self._torch, self._blip2_processor, self._blip2_model

    # ------------------------------------------------------------------
    # Tier 2 — CLIP zero-shot
    # ------------------------------------------------------------------

    def _clip_label(self, image_bytes: bytes) -> Optional[str]:
        """Classify the figure into a chart/figure type label via CLIP."""
        try:
            torch, processor, model = self._load_clip()
            if processor is None or model is None:
                return None
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            device = self._resolve_device()
            inputs = processor(
                text=_CHART_LABELS,
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits_per_image
                probs = logits.softmax(dim=1)
                best_idx = int(probs.argmax(dim=1).item())
            return _CHART_LABELS[best_idx]
        except Exception as exc:
            logger.debug("CLIP labeling failed: %s", exc)
            return None

    def _load_clip(self):
        """Lazy-load CLIP model + processor (cached on instance)."""
        if self._clip_processor is not None:
            return self._torch, self._clip_processor, self._clip_model
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            self._torch = torch
            device = self._resolve_device()
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(device)
            self._clip_model.eval()
            logger.info("CLIP loaded on %s", device)
        except Exception as exc:
            logger.debug("CLIP unavailable: %s", exc)
            self._clip_processor = None
            self._clip_model = None
        return self._torch, self._clip_processor, self._clip_model

    @staticmethod
    def _label_to_caption(label: str, region: Region, doc_title: str) -> str:
        """Turn a CLIP label into a human-readable caption string."""
        page_str = f" on page {region.page_no + 1}" if region.page_no >= 0 else ""
        title_str = f" in {doc_title}" if doc_title else ""
        return f"{label.capitalize()}{page_str}{title_str}"

    # ------------------------------------------------------------------
    # Tier 3 — Metadata fallback (always available)
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_caption(region: Region, doc_title: str) -> str:
        """Build a deterministic caption from region geometry + document title."""
        parts = ["Figure"]
        if region.page_no >= 0:
            parts.append(f"on page {region.page_no + 1}")
        if doc_title:
            parts.append(f"from {doc_title}")
        if region.bbox:
            x0, y0, x1, y1 = region.bbox
            w = round(x1 - x0, 1)
            h = round(y1 - y0, 1)
            parts.append(f"({w}×{h} pts)")
        return " ".join(parts)

    def _resolve_device(self) -> str:
        """Return the torch device string to use."""
        if self._device is not None:
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tripath.utils import get_logger, trace_execution

logger = get_logger(__name__)


class FormatAwareLoader:
    """Extract text from a broad range of document formats for local ingestion.

    Two Docling modes are supported:

    * **fast** (default): text-only export via ``export_to_markdown()``.
      ``do_table_structure=False``, minimal memory use.
    * **deep**: full layout-aware parse with TableFormer enabled.
      Returns the raw ``ConversionResult`` for use by ``DoclingLayoutParser``
      and ``TableSerializer``.  Activate by passing ``deep=True`` to
      :meth:`load_deep`.
    """

    SUPPORTED_EXTENSIONS = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
    }

    @trace_execution(logger=logger)
    def load(self, path: str | Path) -> Dict[str, object]:
        """Load *path* and return a text-payload dict (fast mode)."""
        path = Path(path)
        suffix = path.suffix.lower()
        content_type = self._mime_type(path)
        docling_text = self._load_docling_fast(path) if suffix in {".pdf", ".docx", ".pptx", ".xlsx"} else None
        if docling_text:
            text = docling_text
        elif suffix == ".txt" or suffix == ".md":
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix in {".html", ".htm", ".csv"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            text = self._load_pdf(path)
        elif suffix in {".docx", ".pptx", ".xlsx"}:
            text = self._fallback_text(path, content_type)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""

        return {
            "path": str(path),
            "suffix": suffix,
            "mime_type": content_type,
            "text": text,
            "supported": suffix in self.SUPPORTED_EXTENSIONS or path.exists(),
        }

    def load_deep(self, path: str | Path) -> Optional[Any]:
        """Run Docling in deep layout mode and return the raw ``ConversionResult``.

        Returns ``None`` if Docling is unavailable or the conversion fails.
        The caller (``Phase1Pipeline``) passes this result to
        ``DoclingLayoutParser`` which iterates the typed document items.

        Parameters
        ----------
        path:
            Path to the document.  Only PDF/DOCX/PPTX/XLSX trigger Docling;
            other formats return ``None`` (text-only formats need no layout
            analysis).
        """
        path = Path(path)
        if path.suffix.lower() not in {".pdf", ".docx", ".pptx", ".xlsx"}:
            return None
        return self._load_docling_deep(path)

    def iter_supported_files(self, input_dir: str | Path) -> List[Path]:
        input_dir = Path(input_dir)
        supported = []
        if not input_dir.exists():
            return supported

        priority = {
            ".txt": 0,
            ".md": 1,
            ".html": 2,
            ".htm": 3,
            ".csv": 4,
            ".pdf": 5,
            ".docx": 6,
            ".pptx": 7,
            ".xlsx": 8,
        }

        for path in input_dir.iterdir():
            if path.is_file() and (
                self._mime_type(path) in self.SUPPORTED_EXTENSIONS.values() or path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ):
                supported.append(path)

        return sorted(
            supported,
            key=lambda path: (
                priority.get(path.suffix.lower(), 99),
                path.name.lower(),
            ),
        )

    def _mime_type(self, path: Path) -> str:
        mime, _ = mimetypes.guess_type(str(path))
        return mime or "application/octet-stream"

    def _fallback_text(self, path: Path, content_type: str) -> str:
        if path.exists() and path.stat().st_size == 0:
            return ""
        if content_type.startswith("application/"):
            return f"[Extracted from {path.suffix.lower()} document] {path.name}"
        return path.read_text(encoding="utf-8", errors="ignore")

    def _load_pdf(self, path: Path) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages_text = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pages_text.append(extracted)
            return "\n\n".join(pages_text)
        except Exception:
            return self._fallback_text(path, "application/pdf")

    def _load_docling_fast(self, path: Path) -> str | None:
        """Fast text-only Docling pass (no table structure, minimal RAM)."""
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = False
            pipeline_options.images_scale = 1.0
            pipeline_options.generate_page_images = False
            pipeline_options.page_batch_size = 1

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            result = converter.convert(str(path))
            text = result.document.export_to_markdown()
            return text if text.strip() else None
        except Exception:
            return None

    def _load_docling_deep(self, path: Path) -> Optional[Any]:
        """Deep layout Docling pass — TableFormer enabled, returns ConversionResult."""
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False          # OCR handled by OCRFallback
            pipeline_options.do_table_structure = True  # Enable TableFormer
            pipeline_options.images_scale = 1.0
            pipeline_options.generate_page_images = False
            pipeline_options.page_batch_size = 2     # Memory-safe batch size

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            return converter.convert(str(path))
        except Exception:
            return None

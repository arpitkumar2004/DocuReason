from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Dict, List


class FormatAwareLoader:
    """Extract text from a broad range of document formats for local ingestion."""

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

    def load(self, path: str | Path) -> Dict[str, object]:
        path = Path(path)
        suffix = path.suffix.lower()
        content_type = self._mime_type(path)
        if suffix == ".txt" or suffix == ".md":
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix in {".html", ".htm"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix in {".csv"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix in {".pdf", ".docx", ".pptx", ".xlsx"}:
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

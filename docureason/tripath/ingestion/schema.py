from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Region:
    type: str
    text: str
    start: int = 0
    end: int = 0
    # Page number (0-indexed) where this region appears; -1 when unknown.
    page_no: int = -1
    # Bounding box as (x0, y0, x1, y1) in document units; None when unknown.
    bbox: Optional[Tuple[float, float, float, float]] = None
    # Which parser produced this region: "docling" | "rule" | "ocr"
    layout_source: str = "rule"
    # GFM Markdown table string (populated by TableSerializer for table regions).
    table_markdown: Optional[str] = None
    # Structured JSON schema {"columns": [...], "rows": [[...]]} for SQL generation.
    table_json: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Chunk:
    id: str
    document_id: str
    region_id: str
    modality: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Document:
    id: str
    source: str
    title: str
    regions: List[Region] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

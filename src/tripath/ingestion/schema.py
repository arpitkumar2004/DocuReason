from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Region:
    type: str
    text: str
    start: int = 0
    end: int = 0
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

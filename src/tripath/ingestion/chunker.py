from __future__ import annotations

import re
from typing import Any, Dict, List

from src.tripath.utils import get_logger, trace_execution

from .schema import Chunk, Document, Region

logger = get_logger(__name__)


def build_ancestry_metadata_header(hierarchy: List[str]) -> str:
    """Format ancestry path metadata header e.g. 'Metadata Header: Title > Section > Subhead'."""
    if not hierarchy:
        return ""
    clean_parts = [h.strip() for h in hierarchy if h and h.strip()]
    if not clean_parts:
        return ""
    return f"Metadata Header: {' > '.join(clean_parts)}\n"


class SectionAwareChunker:
    """Create section-aware child chunks with ancestry metadata headers and parent text linking."""

    def __init__(self, chunk_size: int = 512, overlap: int = 80) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    @trace_execution(logger=logger)
    def chunk_document(self, document: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        if not document.regions:
            return chunks

        # 1. Heading-Body Binding: Combine short titles/headings with their child content
        bound_regions = self._bind_headings_to_body(document.regions)

        # 2. Track hierarchy breadcrumbs across regions
        hierarchy: List[str] = [document.title] if document.title else []

        for reg_idx, reg_group in enumerate(bound_regions):
            combined_text = "\n\n".join([r.text.strip() for r in reg_group if r.text.strip()])
            if not combined_text:
                continue

            primary_region = reg_group[0]
            region_type = primary_region.type

            # Update section hierarchy tracking
            if region_type == "title" or combined_text.startswith("#"):
                header_title = re.sub(r"^#+\s*", "", combined_text.splitlines()[0]).strip()
                if header_title:
                    hierarchy = [document.title, header_title] if document.title else [header_title]

            ancestry_header = build_ancestry_metadata_header(hierarchy)
            field_weight = max(self._infer_field_weight(r.type) for r in reg_group)
            windows = self._window_text(combined_text, ancestry_header)

            for win_idx, window in enumerate(windows):
                chunk_id = f"{document.id}-{region_type}-{reg_idx + 1}-{win_idx + 1}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document.id,
                        region_id=f"{document.id}-region-{reg_idx + 1}",
                        modality=self._modality_for_region(region_type),
                        text=window["text"],
                        metadata={
                            "section": " > ".join(hierarchy) if hierarchy else region_type,
                            "ancestry_header": ancestry_header.strip(),
                            "parent_text": combined_text,
                            "field_weight": field_weight,
                            "chunk_index": win_idx,
                            "token_count": window["token_count"],
                            "region_type": region_type,
                        },
                    )
                )

        return chunks

    def _bind_headings_to_body(self, regions: List[Region]) -> List[List[Region]]:
        """Bind short 1-line headings directly to the subsequent paragraph or table region."""
        grouped: List[List[Region]] = []
        idx = 0
        n = len(regions)

        while idx < n:
            curr = regions[idx]
            is_heading = (
                curr.type == "title"
                or (len(curr.text.strip()) < 60 and curr.text.strip().startswith("#"))
                or (len(curr.text.strip()) < 50 and "\n" not in curr.text.strip())
            )

            if is_heading and idx + 1 < n:
                next_reg = regions[idx + 1]
                grouped.append([curr, next_reg])
                idx += 2
            else:
                grouped.append([curr])
                idx += 1

        return grouped

    def _window_text(self, text: str, ancestry_header: str) -> List[Dict[str, Any]]:
        tokens = text.split()
        windows: List[Dict[str, Any]] = []
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            window_tokens = tokens[start:end]
            body_text = " ".join(window_tokens)
            full_text = f"{ancestry_header}{body_text}".strip()

            windows.append({
                "text": full_text,
                "token_count": len(window_tokens),
            })
            if end == len(tokens):
                break
            start += max(1, self.chunk_size - self.overlap)
        return windows

    @staticmethod
    def _infer_field_weight(region_type: str) -> float:
        """Step 3 field weight multipliers: Header 1.0x, Body 2.5x, Table 3.0x."""
        if region_type == "table":
            return 3.0
        if region_type in ("title", "header"):
            return 1.0
        return 2.5

    def _modality_for_region(self, region_type: str) -> str:
        if region_type == "table":
            return "table"
        if region_type == "figure":
            return "vision"
        return "text"

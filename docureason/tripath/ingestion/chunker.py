"""chunker.py — section-aware child chunking with parent linking.

This module is now called by :class:`docureason.pipeline.DocuReasonPipeline`.  Before
that the pipeline emitted **one chunk per region**, which cost three things:

* no parent/child pair, so parent expansion at retrieval time was inert — the
  retriever's ``len(parent_text) > len(text)`` guard could never be true because
  both strings were the same region text;
* no breadcrumb ancestry header in the indexed text;
* no windowing, so a long table's markdown exceeded ``all-MiniLM-L6-v2``'s
  256-token limit and was silently truncated before embedding.  The longest,
  most information-dense chunks were the worst-represented.

Two defects in this module were fixed at the same time:

* ``region_id`` was derived from the *bound-group* index rather than the region
  index, so chunk lineage pointed at ids that do not exist in ``manifest.json``.
  Groups now carry their source region indices and the canonical id is resolved
  through :func:`~docureason.tripath.ingestion.identity.canonical_region_id`.
* the heading heuristic treated *any* line under 50 characters with no newline as
  a heading, so one-line table rows, figure captions and short paragraphs were
  bound to the following region and inherited heading semantics.  A heading now
  requires a real signal: the region type, or a markdown ``#`` prefix.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from docureason.tripath.utils import get_logger, trace_execution

from .identity import canonical_region_id
from .schema import Chunk, Document, Region

logger = get_logger(__name__)

#: Region types that count as a heading for heading/body binding.
HEADING_REGION_TYPES = frozenset({"title", "header", "section_header"})

#: ``chunk_size`` and ``parent_max_words`` are counted in **whitespace-delimited
#: words**, not model tokens.  Word counts run roughly 0.75x token counts for
#: English prose, so the default child size is chosen to stay inside the
#: 256-token limit of the default embedding model.
CHUNK_SIZE_UNIT = "words"


def build_ancestry_metadata_header(hierarchy: List[str]) -> str:
    """Format ancestry path metadata header e.g. 'Metadata Header: Title > Section > Subhead'."""
    if not hierarchy:
        return ""
    clean_parts = [h.strip() for h in hierarchy if h and h.strip()]
    if not clean_parts:
        return ""
    return f"Metadata Header: {' > '.join(clean_parts)}\n"


def is_heading_region(region: Region) -> bool:
    """True when *region* carries an actual heading signal."""
    text = region.text.strip()
    if not text:
        return False
    if region.type in HEADING_REGION_TYPES:
        return True
    # A markdown ATX heading, kept short so a paragraph starting with '#' is not one.
    return text.startswith("#") and len(text.splitlines()[0]) < 120


class SectionAwareChunker:
    """Create section-aware child chunks with ancestry metadata headers and parent text linking.

    Parameters
    ----------
    chunk_size:
        Child window size in words (see :data:`CHUNK_SIZE_UNIT`).
    overlap:
        Words of overlap between consecutive child windows.
    parent_max_words:
        Cap on the parent text carried on each chunk.  Without a cap a very long
        region would push an unbounded string into the generation context budget
        once parent expansion fires.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 80,
        parent_max_words: int = 1024,
    ) -> None:
        self.chunk_size = max(1, int(chunk_size))
        self.overlap = max(0, min(int(overlap), self.chunk_size - 1)) if self.chunk_size > 1 else 0
        self.parent_max_words = max(self.chunk_size, int(parent_max_words))

    @classmethod
    def from_config(cls, config: Any) -> SectionAwareChunker:
        """Build a chunker from ``IngestionConfig`` (or anything carrying ``.ingestion``)."""
        ingestion = getattr(config, "ingestion", config)
        return cls(
            chunk_size=getattr(ingestion, "child_chunk_size", 512),
            overlap=getattr(ingestion, "chunk_overlap", 80),
            parent_max_words=getattr(ingestion, "parent_region_size", 1024),
        )

    @trace_execution(logger=logger)
    def chunk_document(self, document: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        if not document.regions:
            return chunks

        bound_groups = self._bind_headings_to_body(document.regions)
        hierarchy: List[str] = [document.title] if document.title else []

        for group in bound_groups:
            group_regions = [document.regions[i] for i in group]
            combined_text = "\n\n".join(r.text.strip() for r in group_regions if r.text.strip())
            if not combined_text:
                continue

            # The primary region is the group's first *non-heading* member when one
            # exists, so a chunk's lineage points at the content region rather than
            # at the heading that was folded into it.
            primary_index = next(
                (i for i in group if not is_heading_region(document.regions[i])),
                group[0],
            )
            primary_region = document.regions[primary_index]
            region_type = primary_region.type
            region_id = canonical_region_id(document.id, primary_region, primary_index)

            if is_heading_region(group_regions[0]):
                header_title = re.sub(r"^#+\s*", "", group_regions[0].text.strip().splitlines()[0]).strip()
                if header_title:
                    hierarchy = [document.title, header_title] if document.title else [header_title]

            ancestry_header = build_ancestry_metadata_header(hierarchy)
            field_weight = max(self._infer_field_weight(r.type) for r in group_regions)
            parent_text = self._cap_words(combined_text, self.parent_max_words)
            windows = self._window_text(combined_text, ancestry_header)

            for win_idx, window in enumerate(windows):
                chunks.append(
                    Chunk(
                        id=f"{region_id}-chunk-{win_idx}",
                        document_id=document.id,
                        region_id=region_id,
                        modality=self._modality_for_region(region_type),
                        text=window["text"],
                        metadata={
                            "section": " > ".join(hierarchy) if hierarchy else region_type,
                            "ancestry_header": ancestry_header.strip(),
                            "parent_text": parent_text,
                            "field_weight": field_weight,
                            "chunk_index": win_idx,
                            "chunk_count": len(windows),
                            "word_count": window["word_count"],
                            # Retained under the old key for consumers written before
                            # the unit was corrected; both hold a *word* count.
                            "token_count": window["word_count"],
                            "chunk_size_unit": CHUNK_SIZE_UNIT,
                            "region_type": region_type,
                            "region_id": region_id,
                            "bound_region_indices": list(group),
                        },
                    )
                )

        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _bind_headings_to_body(regions: List[Region]) -> List[List[int]]:
        """Group region *indices*, binding a heading to the region that follows it.

        Returning indices rather than regions is what lets the caller resolve the
        canonical ``region_id``; the previous version returned region objects and the
        chunk id was built from the group ordinal instead.
        """
        grouped: List[List[int]] = []
        idx = 0
        n = len(regions)

        while idx < n:
            if is_heading_region(regions[idx]) and idx + 1 < n and not is_heading_region(regions[idx + 1]):
                grouped.append([idx, idx + 1])
                idx += 2
            else:
                grouped.append([idx])
                idx += 1

        return grouped

    def _window_text(self, text: str, ancestry_header: str) -> List[Dict[str, Any]]:
        words = text.split()
        windows: List[Dict[str, Any]] = []
        start = 0
        stride = max(1, self.chunk_size - self.overlap)

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            window_words = words[start:end]
            body_text = " ".join(window_words)
            windows.append(
                {
                    "text": f"{ancestry_header}{body_text}".strip(),
                    "word_count": len(window_words),
                }
            )
            if end == len(words):
                break
            start += stride

        if not windows:
            windows.append({"text": ancestry_header.strip(), "word_count": 0})
        return windows

    @staticmethod
    def _cap_words(text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + " ...[parent truncated]"

    @staticmethod
    def _infer_field_weight(region_type: str) -> float:
        """Step 3 field weight multipliers: Header 1.0x, Body 2.5x, Table 3.0x."""
        if region_type == "table":
            return 3.0
        if region_type in ("title", "header", "section_header"):
            return 1.0
        return 2.5

    @staticmethod
    def _modality_for_region(region_type: str) -> str:
        if region_type == "table":
            return "table"
        if region_type == "figure":
            return "vision"
        return "text"

from __future__ import annotations

from typing import List, Dict, Any

from .schema import Chunk, Document, Region


class SectionAwareChunker:
    """Create section-aware chunks with token windows and overlap."""

    def __init__(self, chunk_size: int = 128, overlap: int = 24) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        for region in document.regions:
            if not region.text.strip():
                continue
            section_name = self._infer_section(region)
            windows = self._window_text(region.text, section_name)
            for index, window in enumerate(windows):
                chunk_id = f"{document.id}-{region.type}-{index + 1}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document.id,
                        region_id=f"{document.id}-region-{index + 1}",
                        modality=self._modality_for_region(region.type),
                        text=window["text"],
                        metadata={
                            "section": section_name,
                            "chunk_index": index,
                            "token_count": window["token_count"],
                            "region_type": region.type,
                        },
                    )
                )
        return chunks

    def _window_text(self, text: str, section_name: str) -> List[Dict[str, Any]]:
        tokens = text.split()
        windows: List[Dict[str, Any]] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            window_tokens = tokens[start:end]
            window_text = " ".join(window_tokens)
            windows.append({
                "text": window_text,
                "token_count": len(window_tokens),
                "section": section_name,
            })
            if end == len(tokens):
                break
            start += max(1, self.chunk_size - self.overlap)
        return windows

    def _infer_section(self, region: Region) -> str:
        if region.type == "title":
            return "title"
        if region.type == "table":
            return "table"
        if region.type == "figure":
            return "figure"
        return "body"

    def _modality_for_region(self, region_type: str) -> str:
        if region_type == "table":
            return "table"
        if region_type == "figure":
            return "vision"
        return "text"

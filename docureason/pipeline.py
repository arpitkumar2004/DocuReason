from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class Phase1Pipeline:
    """Build a lightweight corpus and index artifact from sample text documents."""

    def __init__(self, input_dir: str | Path, output_dir: str | Path) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, object]:
        documents: List[Dict[str, object]] = []
        chunks: List[Dict[str, object]] = []

        for path in sorted(self.input_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            doc_id = f"doc-{path.stem}"
            regions = self._segment_regions(text)
            doc_regions = []
            doc_chunks = []
            for index, region in enumerate(regions):
                region_id = f"{doc_id}-region-{index + 1}"
                modality = self._modality_for_region(region["type"])
                metadata = {
                    "source_path": str(path.name),
                    "region_type": region["type"],
                    "modality": modality,
                }
                doc_regions.append({
                    "id": region_id,
                    "type": region["type"],
                    "text": region["text"],
                    "start": region["start"],
                    "end": region["end"],
                    "metadata": metadata,
                })
                chunk = {
                    "id": f"{region_id}-chunk",
                    "document_id": doc_id,
                    "region_id": region_id,
                    "modality": modality,
                    "text": region["text"],
                    "metadata": metadata,
                }
                chunks.append(chunk)
                doc_chunks.append(chunk)

            documents.append({
                "id": doc_id,
                "source": str(path.name),
                "title": self._title_from_text(text),
                "regions": doc_regions,
                "chunks": doc_chunks,
                "metadata": {
                    "document_id": doc_id,
                    "source_path": str(path.name),
                    "chunk_count": len(doc_chunks),
                },
            })

        corpus = {"documents": documents}
        index = {"chunks": chunks}
        audit = {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "notes": [
                "Basic ingestion pipeline completed.",
                "Table and figure regions are captured as structured metadata.",
            ],
        }

        (self.output_dir / "corpus.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")
        (self.output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        (self.output_dir / "quality_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

        return {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "output_dir": str(self.output_dir),
        }

    def _title_from_text(self, text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.lower().startswith(("table:", "figure:")):
                return stripped
        return "Untitled"

    def _modality_for_region(self, region_type: str) -> str:
        if region_type == "table":
            return "table"
        if region_type == "figure":
            return "vision"
        return "text"

    def _segment_regions(self, text: str) -> List[Dict[str, object]]:
        lines = [line.rstrip() for line in text.splitlines()]
        regions: List[Dict[str, object]] = []
        if not lines:
            return regions

        title = self._title_from_text(text)
        if title:
            regions.append({"type": "title", "text": title, "start": 0, "end": len(title)})

        body_lines: List[str] = []
        current_type = "body"
        current_lines: List[str] = []
        current_start = 0

        def flush_current() -> None:
            nonlocal current_lines, current_type, current_start
            if current_lines:
                text_block = "\n".join(current_lines).strip()
                if text_block:
                    regions.append({
                        "type": current_type,
                        "text": text_block,
                        "start": current_start,
                        "end": current_start + len(text_block),
                    })
                current_lines = []
                current_type = "body"
                current_start = 0

        for index, line in enumerate(lines[1:], start=1):
            stripped = line.strip()
            if not stripped:
                if current_lines:
                    flush_current()
                continue

            lower = stripped.lower()
            if lower.startswith("table:"):
                flush_current()
                current_type = "table"
                current_lines = [stripped]
                current_start = index
            elif lower.startswith("figure:"):
                flush_current()
                current_type = "figure"
                current_lines = [stripped]
                current_start = index
            else:
                if not current_lines:
                    current_start = index
                current_lines.append(stripped)

        flush_current()
        return regions

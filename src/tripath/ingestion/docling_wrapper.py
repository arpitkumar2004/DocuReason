from __future__ import annotations

from pathlib import Path
from typing import List

from .schema import Chunk, Document, Region


class DoclingWrapper:
    """A thin adapter over the document ingestion pipeline."""

    def __init__(self, input_dir: str | Path, output_dir: str | Path) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

    def load_corpus(self) -> List[Document]:
        documents: List[Document] = []
        corpus_path = self.output_dir / "corpus.json"
        if corpus_path.exists():
            import json
            payload = json.loads(corpus_path.read_text(encoding="utf-8"))
            for item in payload.get("documents", []):
                regions = [
                    Region(
                        type=region.get("type", "body"),
                        text=region.get("text", ""),
                        start=region.get("start", 0),
                        end=region.get("end", 0),
                        metadata=region.get("metadata"),
                    )
                    for region in item.get("regions", [])
                ]
                chunks = [
                    Chunk(
                        id=chunk.get("id", ""),
                        document_id=chunk.get("document_id", item.get("id", "")),
                        region_id=chunk.get("region_id", ""),
                        modality=chunk.get("modality", "text"),
                        text=chunk.get("text", ""),
                        metadata=chunk.get("metadata"),
                    )
                    for chunk in item.get("chunks", [])
                ]
                documents.append(
                    Document(
                        id=item.get("id", ""),
                        source=item.get("source", ""),
                        title=item.get("title", ""),
                        regions=regions,
                        chunks=chunks,
                        metadata=item.get("metadata"),
                    )
                )
        return documents

    def ingest(self, force_reingest: bool = False) -> List[Document]:
        corpus_path = self.output_dir / "corpus.json"
        if not force_reingest and corpus_path.exists():
            return self.load_corpus()

        from docureason.pipeline import DocuReasonPipeline
        pipeline = DocuReasonPipeline(input_dir=self.input_dir, output_dir=self.output_dir)
        pipeline.run()
        return self.load_corpus()

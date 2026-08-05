from src.tripath.ingestion.chunker import SectionAwareChunker, build_ancestry_metadata_header
from src.tripath.ingestion.schema import Document, Region
from src.tripath.retrieval.ranker import Ranker


def test_build_ancestry_metadata_header():
    assert build_ancestry_metadata_header(["Annual Report", "Balance Sheet", "Assets"]) == "Metadata Header: Annual Report > Balance Sheet > Assets\n"
    assert build_ancestry_metadata_header([]) == ""


def test_heading_body_binding_and_breadcrumbs():
    doc = Document(
        id="doc1",
        title="Financial Statement",
        source="doc1.pdf",
        regions=[
            Region(type="title", text="# Executive Summary", bbox=(0, 0, 100, 20)),
            Region(type="paragraph", text="This is the full executive summary detailing total annual assets.", bbox=(0, 20, 100, 80)),
        ]
    )
    chunker = SectionAwareChunker(chunk_size=128)
    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) >= 1
    # Check that ancestry metadata header is injected
    assert "Metadata Header: Financial Statement > Executive Summary" in chunks[0].text
    # Check that short heading is bound to paragraph text (not isolated)
    assert "total annual assets" in chunks[0].text


def test_ranker_penalizes_isolated_short_headings():
    ranker = Ranker()
    candidates = [
        {"id": "c1", "text": "Metadata Header: Report\nShort Title", "modality": "text", "score": 10.0},
        {"id": "c2", "text": "Metadata Header: Report\nDetailed paragraph explaining full financial asset values and revenue growth for 2019.", "modality": "text", "score": 8.0},
    ]
    ranked = ranker.rank("financial asset values revenue", candidates)
    # The detailed paragraph should outrank the short standalone heading
    assert ranked[0]["id"] == "c2"

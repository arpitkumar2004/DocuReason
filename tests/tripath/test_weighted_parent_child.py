from src.tripath.ingestion.chunker import SectionAwareChunker, build_ancestry_metadata_header
from src.tripath.ingestion.schema import Document, Region
from src.tripath.retrieval.text_retrieval import TextRetrieval
from src.tripath.retrieval.table_retrieval import TableRetrieval
from src.tripath.retrieval.ranker import Ranker


def test_build_ancestry_metadata_header():
    header = build_ancestry_metadata_header(["Balance Sheet", "Guidance", "Currency Segregation"])
    assert header == "Metadata Header: Balance Sheet > Guidance > Currency Segregation\n"


def test_ancestry_path_and_field_weighting_in_chunker():
    doc = Document(
        id="doc1",
        title="Statement of Financial Position",
        source="statement.pdf",
        regions=[
            Region(type="title", text="# Guidance", bbox=(0, 0, 100, 20)),
            Region(type="paragraph", text="This segregation into foreign currency and domestic currency is detailed below.", bbox=(0, 20, 100, 80)),
        ]
    )
    chunker = SectionAwareChunker(chunk_size=128)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    # Check Metadata Header prefix (Step 1)
    assert "Metadata Header: Statement of Financial Position > Guidance" in chunks[0].text
    # Check field weight assignment (Step 3)
    assert chunks[0].metadata["field_weight"] == 2.5


def test_weighted_text_and_table_retrieval():
    text_retriever = TextRetrieval()
    table_retriever = TableRetrieval()
    doc = Document(
        id="doc1",
        title="Report",
        source="report.pdf",
        regions=[
            Region(type="title", text="General Guidance", bbox=(0, 0, 100, 20)),
            Region(type="paragraph", text="Detailed segregation of foreign currency assets.", bbox=(0, 20, 100, 50)),
            Region(type="table", text="Table: Foreign Currency Assets $500M", bbox=(0, 50, 100, 100)),
        ]
    )
    text_results = text_retriever.retrieve("foreign currency assets", [doc])
    table_results = table_retriever.retrieve("foreign currency assets", [doc])

    assert len(text_results) >= 1
    # Paragraph should get 2.5x field weight multiplier
    assert text_results[0]["field_weight"] == 2.5
    assert len(table_results) >= 1
    # Table should get 3.0x field weight multiplier
    assert table_results[0]["field_weight"] == 3.0


def test_ranker_boosts_body_and_table_over_titles():
    ranker = Ranker()
    candidates = [
        {"id": "c1", "text": "Metadata Header: Report > Guidance\nGeneral Guidance", "modality": "text", "score": 1.0},
        {"id": "c2", "text": "Metadata Header: Report > Assets\nThis segregation into foreign currency and domestic currency details asset growth.", "modality": "text", "score": 2.5},
    ]
    ranked = ranker.rank("foreign currency segregation", candidates)
    # Paragraph content should outrank the isolated general title
    assert ranked[0]["id"] == "c2"

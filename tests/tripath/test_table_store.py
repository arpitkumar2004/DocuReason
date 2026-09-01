"""test_table_store.py — DuckDB tables registered at ingestion, not per query.

Before the store, every table was re-created and re-populated on every query, and its
column types were re-inferred each time - so the same column could be DECIMAL for one
question and VARCHAR for the next. These tests pin the registration contract and the
fallback that keeps a partially built store from breaking retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docureason.tripath.indexing.table_store import DuckDBTableStore, table_name_for
from docureason.tripath.ingestion.schema import Document, Region
from docureason.tripath.retrieval.table_sql import TableSQLRetriever

pytestmark = pytest.mark.unit


def _table_region(region_id: str, columns, rows, page_no: int = 1) -> Region:
    region = Region(type="table", text="table", start=0, end=5, layout_source="docling")
    region.table_json = {"columns": columns, "rows": rows}
    region.metadata = {"region_id": region_id, "page_no": page_no}
    return region


def _document(*regions: Region, doc_id: str = "doc-fixture") -> Document:
    return Document(id=doc_id, source="f.html", title="F", regions=list(regions), metadata={})


@pytest.fixture
def registered(tmp_path: Path):
    document = _document(
        _table_region(
            "doc-fixture-region-0", ["segment", "revenue_2024"], [["Americas", "15000"], ["Europe", "25000"]]
        ),
        _table_region("doc-fixture-region-1", ["metric", "value"], [["Total assets", "9100"]]),
    )
    store = DuckDBTableStore(tmp_path)
    stats = store.build([document])
    yield document, store, stats
    store.close()


def test_every_table_region_is_registered(registered) -> None:
    _, store, stats = registered
    assert stats.tables_registered == 2
    assert stats.rows_registered == 3
    assert stats.regions_skipped == 0
    assert set(store.index()) == {"doc-fixture-region-0", "doc-fixture-region-1"}


def test_types_are_decided_once_at_ingestion(registered) -> None:
    _, store, _ = registered
    entry = store.lookup("doc-fixture-region-0")
    assert entry.column_types == {"segment": "VARCHAR", "revenue_2024": "INTEGER"}
    assert entry.row_count == 2
    assert entry.page_no == 1


def test_year_like_values_stay_textual(tmp_path: Path) -> None:
    """A bare four-digit value is a year as often as a number, so it stays VARCHAR.

    Worth pinning: it is the reason a column of small round figures does not arrive as
    an INTEGER, which is surprising until you know it is deliberate.
    """
    store = DuckDBTableStore(tmp_path)
    store.build([_document(_table_region("doc-y-region-0", ["k", "v"], [["a", "1500"], ["b", "2500"]]))])
    try:
        assert store.lookup("doc-y-region-0").column_types["v"] == "VARCHAR"
    finally:
        store.close()


def test_the_index_is_written_beside_the_database(registered, tmp_path: Path) -> None:
    """The index is the contract; without it on disk the store is unusable."""
    assert (tmp_path / "tables.duckdb").exists()
    payload = json.loads((tmp_path / "tables_index.json").read_text(encoding="utf-8"))
    assert payload["doc-fixture-region-1"]["table_name"] == table_name_for("doc-fixture-region-1")


def test_prepared_tables_are_readable_and_hold_the_rows(registered) -> None:
    _, store, _ = registered
    connection = store.connection()
    name = store.lookup("doc-fixture-region-0").table_name
    assert connection.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0] == 2


def test_the_retriever_uses_the_prepared_table(registered) -> None:
    document, store, _ = registered
    retriever = TableSQLRetriever(store=store)
    results = retriever.retrieve("revenue in 2024", [document])

    assert results, "the prepared path returned nothing"
    assert results[0]["mode"] == "prepared_duckdb_sql"
    assert retriever.last_stats == {"prepared_tables": 2, "in_memory_tables": 0}


def test_an_absent_store_falls_back_to_the_in_memory_path(tmp_path: Path) -> None:
    """A corpus ingested before the store existed must still answer table queries."""
    document = _document(_table_region("doc-fixture-region-0", ["segment", "revenue_2024"], [["Americas", "1500"]]))
    retriever = TableSQLRetriever(store=DuckDBTableStore(tmp_path / "never-built"))
    results = retriever.retrieve("revenue in 2024", [document])

    assert results[0]["mode"] in ("typed_duckdb_sql", "sql_simulation")
    assert retriever.last_stats["prepared_tables"] == 0


def test_unregisterable_regions_are_counted_by_reason(tmp_path: Path) -> None:
    no_json = Region(type="table", text="Table: something", start=0, end=5, layout_source="rule")
    no_json.metadata = {"region_id": "doc-fixture-region-9"}
    empty = _table_region("doc-fixture-region-8", ["a", "b"], [])
    duplicate = _table_region("doc-fixture-region-7", ["a", "a"], [["1", "2"]])

    store = DuckDBTableStore(tmp_path)
    stats = store.build([_document(no_json, empty, duplicate)])
    store.close()

    assert stats.tables_registered == 0
    assert stats.skipped_reasons == {"no_table_json": 1, "no_rows": 1, "duplicate_columns": 1}


def test_a_rebuild_does_not_leave_the_previous_corpus_reachable(tmp_path: Path) -> None:
    """A stale table would answer with another document's numbers."""
    first = _document(_table_region("doc-a-region-0", ["k", "v"], [["x", "1"]]), doc_id="doc-a")
    second = _document(_table_region("doc-b-region-0", ["k", "v"], [["y", "2"]]), doc_id="doc-b")

    store = DuckDBTableStore(tmp_path)
    store.build([first])
    store.close()

    rebuilt = DuckDBTableStore(tmp_path)
    rebuilt.build([second])
    try:
        assert set(rebuilt.index()) == {"doc-b-region-0"}
        connection = rebuilt.connection()
        names = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        assert names == {rebuilt.lookup("doc-b-region-0").table_name}
    finally:
        rebuilt.close()


def test_table_names_are_deterministic_and_sql_safe() -> None:
    assert table_name_for("doc-a b/c-region-1") == "doc_a_b_c_region_1"
    assert table_name_for("doc-x-region-0") == table_name_for("doc-x-region-0")
    assert table_name_for("9lives").startswith("t_"), "an identifier cannot start with a digit"
    assert table_name_for("") == "unnamed"

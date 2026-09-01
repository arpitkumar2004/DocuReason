"""test_security_boundaries.py — the two request-controlled boundaries.

Audit §10 blind spot #11, covering findings S2/S3 (path traversal and arbitrary
write through ``input_dir``/``output_dir``) and S4 (SQL identifier injection from a
document's own table header).  Both were reachable from routes that have no
authentication, and neither had a test.

What is *not* claimed here: S1 is still open — these routes are unauthenticated, so
everything inside the deployment root remains reachable by any caller.  These tests
pin containment, not authorization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from docureason.tripath.retrieval.table_sql import TableSQLRetriever, quote_identifier
from docureason.tripath.serving import main as serving_main

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# S2 / S3 — request-supplied paths stay inside the deployment root
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate",
    [
        "../..",
        "../../etc",
        "samples/../../..",
        "./../..",
        "/etc/passwd",
        r"\Windows\System32",
        "C:/Windows",
        r"C:\Windows\System32",
        "..",
    ],
)
def test_paths_outside_the_root_are_rejected(candidate: str) -> None:
    with pytest.raises(serving_main.CorpusPathError):
        serving_main.resolve_within_root(candidate, "samples")


@pytest.mark.parametrize("candidate", ["samples", "./samples", "samples/../samples", "artifacts/nested/run"])
def test_relative_paths_inside_the_root_are_accepted(candidate: str) -> None:
    resolved = serving_main.resolve_within_root(candidate, "samples")
    root = serving_main.ROOT.resolve()
    assert resolved == root or root in resolved.parents


def test_an_empty_value_falls_back_to_the_default() -> None:
    assert serving_main.resolve_within_root(None, "samples") == (serving_main.ROOT / "samples").resolve()
    assert serving_main.resolve_within_root("   ", "samples") == (serving_main.ROOT / "samples").resolve()


def test_a_symlink_pointing_out_of_the_root_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Containment is checked after resolution, so a link cannot tunnel out."""
    root = tmp_path / "deployment"
    (root / "samples").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "escape").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):  # Windows without developer mode
        pytest.skip("symlink creation not permitted on this platform")

    monkeypatch.setattr(serving_main, "ROOT", root)
    with pytest.raises(serving_main.CorpusPathError):
        serving_main.resolve_within_root("escape", "samples")


class _StubService:
    """Never reached by a rejected path; present so a 400 cannot be an ingestion error."""

    def __init__(self, *_: Any, **__: Any) -> None:
        raise AssertionError("a rejected path must not construct a QueryService")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(serving_main, "QueryService", _StubService)
    serving_main._SERVICES.clear()
    # Deliberately not used as a context manager: entering it would run the lifespan
    # warm-up, which ingests the default corpus for real.
    yield TestClient(serving_main.app)
    serving_main._SERVICES.clear()


def test_query_route_rejects_a_traversal_payload(client) -> None:
    response = client.post("/api/query", json={"query": "revenue", "input_dir": "../../.."})
    assert response.status_code == 400
    assert "root" in response.json()["detail"].lower()


def test_ingest_route_rejects_a_traversal_output_dir(client) -> None:
    response = client.post("/api/ingest", json={"input_dir": "samples", "output_dir": "../../tmp/pwned"})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# S4 — a table header cannot break out of its SQL identifier
# ---------------------------------------------------------------------------

#: A header cell as it would arrive from a parsed document table.
EVIL_HEADER = 'revenue_2024" ; DROP TABLE doc_table; --'


def _evil_table() -> dict:
    return {"columns": ["region", EVIL_HEADER], "rows": [["North", "1500"], ["South", "2500"]]}


def test_quote_identifier_doubles_embedded_quotes() -> None:
    assert quote_identifier(EVIL_HEADER) == '"revenue_2024"" ; DROP TABLE doc_table; --"'
    assert quote_identifier("Revenue") == '"Revenue"'
    assert quote_identifier(None) == '""'
    assert "\x00" not in quote_identifier("bad\x00name")


@pytest.mark.parametrize("query", ["total sum of revenue", "revenue in 2024", "total assets in 2024"])
def test_crafted_header_stays_inside_one_identifier(query: str) -> None:
    sql, _, _ = TableSQLRetriever()._execute_sql_query(query, _evil_table())

    assert '""' in sql, "the embedded quote was not escaped"
    assert sql.count('"') % 2 == 0, f"unbalanced identifier quoting: {sql}"


def test_the_injected_statement_does_not_execute() -> None:
    """Run the generated SQL for real and assert the table survives it."""
    duckdb = pytest.importorskip("duckdb")
    sql, results, score = TableSQLRetriever()._execute_sql_query("revenue in 2024", _evil_table())

    con = duckdb.connect(database=":memory:")
    con.execute(f'CREATE TABLE doc_table ("region" VARCHAR, {quote_identifier(EVIL_HEADER)} VARCHAR)')
    con.execute("INSERT INTO doc_table VALUES (?, ?)", ["North", "1500"])
    con.execute(sql)

    survivors = con.execute("SELECT count(*) FROM doc_table").fetchone()[0]
    con.close()

    assert survivors == 1, "the injected DROP TABLE executed"
    # And the retriever's own execution returned the table's data, not an error path.
    assert len(results) == 2
    assert score > 0.0


def test_a_crafted_cell_value_is_bound_as_data() -> None:
    """Cell values go through DuckDB placeholders, so they are data, never SQL."""
    table = {
        "columns": ["region", "revenue_2024"],
        "rows": [["North'); DROP TABLE doc_table; --", "1500"]],
    }
    _, results, _ = TableSQLRetriever()._execute_sql_query("revenue in 2024", table)

    assert results, "the crafted cell aborted execution instead of being bound"
    assert results[0]["region"] == "North'); DROP TABLE doc_table; --"

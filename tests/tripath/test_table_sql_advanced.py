from src.tripath.evaluation.table_eval import TableEvaluator
from src.tripath.ingestion.table_serializer import (
    TableSerializer,
    build_header_hierarchy,
    extract_unit_multiplier,
    flatten_column_name,
    normalize_numeric_cell,
)
from src.tripath.retrieval.table_sql import TableSQLRetriever, infer_sql_type


def test_build_header_hierarchy_and_flattening():
    header_rows = [
        [("Revenue by Segment", 2), ("Operating Expenses", 1)],
        [("Q1 2024", 1), ("Q2 2024", 1), ("Total", 1)],
    ]
    paths = build_header_hierarchy(header_rows, n_cols=3)
    assert len(paths) == 3
    assert paths[0] == ("Revenue by Segment", "Q1 2024")
    assert paths[1] == ("Revenue by Segment", "Q2 2024")
    assert paths[2] == ("Operating Expenses", "Total")

    flat0 = flatten_column_name(paths[0])
    assert flat0 == "revenue_by_segment_q1_2024"


def test_numeric_accounting_normalization():
    assert normalize_numeric_cell("(1,234.50)") == -1234.5
    assert normalize_numeric_cell("$5,000") == 5000
    assert normalize_numeric_cell("12.5%") == 12.5
    assert normalize_numeric_cell("N/A") == "N/A"


def test_unit_multiplier_extraction():
    assert extract_unit_multiplier("Statement of Position (in thousands)") == 1000
    assert extract_unit_multiplier("Revenue report (in millions)") == 1000000
    assert extract_unit_multiplier("Standard table") == 1


def test_infer_sql_type():
    assert infer_sql_type(["100", "200", "300"]) == "INTEGER"
    assert infer_sql_type(["100.50", "(50.25)", "300.00"]) == "DECIMAL(18,2)"
    assert infer_sql_type(["North", "South", "East"]) == "VARCHAR"


def test_typed_duckdb_execution_and_self_consistency():
    retriever = TableSQLRetriever()
    table_data = {
        "columns": ["region", "revenue_q1"],
        "rows": [["North", 1500.0], ["South", 2500.0]],
    }
    sql, result, score = retriever._execute_sql_query("revenue total sum", table_data)
    assert "SUM" in sql or "revenue_q1" in sql
    assert score > 0.5
    assert len(result) >= 1


def test_duckdb_empty_string_conversion_handling():
    retriever = TableSQLRetriever()
    # Table containing empty string cells "" in an INTEGER / DECIMAL column
    table_data = {
        "columns": ["description", "note", "2019", "2018"],
        "rows": [
            ["ASSETS", "", "", ""],
            ["Cash and cash equivalents", "4", "1250", "1100"],
            ["Total Assets", "", "5000", "4500"],
        ],
    }
    sql, result, score = retriever._execute_sql_query("What was the total asset value as of 2019?", table_data)
    assert score > 0.0
    assert len(result) >= 1


def test_table_evaluator_teds():
    evaluator = TableEvaluator()
    pred = {"columns": ["col_1", "col_2"], "rows": [["a", "b"]]}
    gt = {"columns": ["col_1", "col_2"], "rows": [["a", "b"]]}
    metrics = evaluator.evaluate_structure_and_content(pred, gt)
    assert metrics["teds_structural_similarity"] == 1.0
    assert metrics["content_accuracy"] == 1.0


def test_stub_header_realignment_and_sanitization():
    # Test shifted fallback header: ["note", "2019", "2018", "col_4"] -> ["description", "note", "2019", "2018"]
    columns = ["note", "2019", "2018", "col_4"]
    rows = [["ASSETS", "", "", ""], ["Cash", "1", "500", "400"]]

    realigned_cols, _ = TableSerializer._sanitize_stub_columns_and_headers(columns, rows)
    assert realigned_cols[0] == "description"
    assert realigned_cols[1:] == ["note", "2019", "2018"]

    md = TableSerializer._to_markdown(realigned_cols, rows)
    assert "| description | note | 2019 | 2018 |" in md
    assert "| ASSETS |" in md


def test_spatial_cell_sorting():
    class DummyCell:
        def __init__(self, x0, y0, text):
            self.bbox = type("BBox", (), {"x0": x0, "y0": y0})()
            self.text = text

    c1 = DummyCell(200.0, 10.0, "2019")
    c2 = DummyCell(20.0, 10.0, "ASSETS")
    c3 = DummyCell(300.0, 10.0, "2018")

    sorted_cells = TableSerializer._sort_cells_spatially([c1, c2, c3])
    assert [c.text for c in sorted_cells] == ["ASSETS", "2019", "2018"]

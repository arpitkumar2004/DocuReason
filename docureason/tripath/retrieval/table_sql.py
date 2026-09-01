from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docureason.tripath.utils import get_logger, trace_execution

from ..ingestion.identity import canonical_region_id
from ..ingestion.schema import Document

logger = get_logger(__name__)
FAILURE_LOG_PATH = Path("artifacts/sql_failure_log.jsonl")


def quote_identifier(name: Any) -> str:
    """Quote *name* as a SQL identifier, escaping any embedded double quote.

    Column names here come from document content - a parsed table header - and were
    interpolated straight into CREATE TABLE and SELECT as a quoted identifier.  A
    header cell containing a double quote therefore closed the identifier and the
    remainder of the cell was parsed as SQL, in a DuckDB session reachable from the
    unauthenticated /api/query route (audit finding S4).  Doubling the quote is the
    SQL-standard escape and keeps the identifier one token whatever the document
    contains.
    """
    text = "" if name is None else str(name)
    # A NUL byte terminates the string for some client libraries, so it cannot be
    # escaped into safety - drop it.
    text = text.replace("\x00", "")
    escaped = text.replace('"', '""')
    return f'"{escaped}"'


def infer_sql_type(values: List[Any]) -> str:
    """Lever A: Infer DuckDB data type (DECIMAL(18,2), INTEGER, DATE, or VARCHAR)."""
    cleaned = []
    for v in values:
        if v is None:
            continue
        v_str = str(v).replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("%", "").strip("()")
        if v_str and v_str not in ("-", "—", "N/A", "null", "None", "nan"):
            cleaned.append(v_str)

    if not cleaned:
        return "VARCHAR"

    # Check for DATE format (YYYY-MM-DD or YYYY)
    if all(re.fullmatch(r"\d{4}(-\d{2}-\d{2})?", v) for v in cleaned):
        return "VARCHAR"

    # Check for INTEGER
    if all(re.fullmatch(r"-?\d+", v) for v in cleaned):
        return "INTEGER"

    # Check for DECIMAL
    if all(re.fullmatch(r"-?\d+\.?\d*", v) for v in cleaned):
        return "DECIMAL(18,2)"

    return "VARCHAR"


def sanitize_cell_for_sql(val: Any, col_type: str) -> Any:
    """Sanitize cell value for DuckDB typed insertion (convert empty strings to None / NULL)."""
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str or val_str in ("-", "—", "N/A", "null", "None", "nan"):
        return None if col_type in ("INTEGER", "DECIMAL(18,2)") else val_str

    if col_type in ("INTEGER", "DECIMAL(18,2)"):
        cleaned = val_str.replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("%", "").strip()
        is_neg = False
        if cleaned.startswith("(") and cleaned.endswith(")"):
            is_neg = True
            cleaned = cleaned[1:-1].strip()
        try:
            if col_type == "INTEGER":
                num = int(float(cleaned))
                return -num if is_neg else num
            else:
                num = float(cleaned)
                return -num if is_neg else num
        except (ValueError, TypeError):
            return None

    return val_str


def log_sql_failure(query: str, generated_sql: str, error: str, table_data: Dict[str, Any]) -> None:
    """Lever D: Data flywheel failure logging for offline LoRA fine-tuning curation."""
    try:
        FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "query": query,
            "generated_sql": generated_sql,
            "error": error,
            "columns": table_data.get("columns", []),
            "sample_rows": table_data.get("rows", [])[:3],
        }
        with open(FAILURE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to log SQL failure: %s", exc)


class TableSQLRetriever:
    """Advanced DuckDB SQL execution engine with typed schemas and self-consistency."""

    def __init__(self, config: Optional[Any] = None, store: Optional[Any] = None) -> None:
        self.config = config
        self._duckdb_available: Optional[bool] = None
        #: Optional :class:`~docureason.tripath.indexing.table_store.DuckDBTableStore`.
        #: When a region is registered there, its table already exists with types
        #: decided at ingestion, so the query does not re-create and re-insert it.
        self.store = store
        #: Counters for the most recent retrieve(), so the response can say which path
        #: served each table rather than leaving the two indistinguishable.
        self.last_stats: Dict[str, int] = {"prepared_tables": 0, "in_memory_tables": 0}

    def _has_duckdb(self) -> bool:
        if self._duckdb_available is None:
            try:
                import duckdb  # noqa: F401

                self._duckdb_available = True
            except ImportError:
                self._duckdb_available = False
                logger.warning("duckdb package not installed — using in-memory Python SQL runner")
        return self._duckdb_available

    @trace_execution(logger=logger)
    def retrieve(self, query: str, documents: List[Document]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        lowered_query = query.lower()
        self.last_stats = {"prepared_tables": 0, "in_memory_tables": 0}

        for document in documents:
            for idx, region in enumerate(document.regions):
                if region.type != "table":
                    continue

                table_data = self._extract_table_data(region)
                if not table_data or not table_data.get("columns"):
                    continue

                region_id = canonical_region_id(document.id, region, idx)
                prepared = self._prepared_table(region_id)
                if prepared is not None:
                    self.last_stats["prepared_tables"] += 1
                    sql_query, sql_result, score = self._execute_sql_query(
                        lowered_query,
                        table_data,
                        table_name=prepared.table_name,
                        connection=self.store.connection(),
                        column_types=prepared.column_types,
                    )
                else:
                    self.last_stats["in_memory_tables"] += 1
                    sql_query, sql_result, score = self._execute_sql_query(lowered_query, table_data)

                if score > 0.0:
                    results.append(
                        {
                            "document_id": document.id,
                            "region_id": region_id,
                            "score": round(score, 3),
                            "text": region.table_markdown or region.text,
                            "modality": "table",
                            "mode": self._mode_label(prepared is not None),
                            "sql_query": sql_query,
                            "sql_result": sql_result,
                            "table_schema": table_data.get("columns", []),
                            "inferred_types": table_data.get("inferred_types", {}),
                            "unit_multiplier": table_data.get("unit_multiplier", 1),
                            "query": query,
                        }
                    )

        return sorted(results, key=lambda item: item["score"], reverse=True)

    def _prepared_table(self, region_id: str) -> Optional[Any]:
        """The ingestion-registered table for *region_id*, when the store has one."""
        if self.store is None or not getattr(self.store, "available", False):
            return None
        entry = self.store.lookup(region_id)
        if entry is None or self.store.connection() is None:
            return None
        return entry

    def _mode_label(self, prepared: bool) -> str:
        if prepared:
            return "prepared_duckdb_sql"
        return "typed_duckdb_sql" if self._has_duckdb() else "sql_simulation"

    def _extract_table_data(self, region: Any) -> Optional[Dict[str, Any]]:
        if getattr(region, "table_json", None):
            if isinstance(region.table_json, dict):
                return region.table_json
            if isinstance(region.table_json, str):
                try:
                    return json.loads(region.table_json)
                except Exception:
                    pass

        metadata = getattr(region, "metadata", {}) or {}
        if "table_json" in metadata:
            raw = metadata["table_json"]
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except Exception:
                    pass

        text = getattr(region, "table_markdown", "") or getattr(region, "text", "")
        return self._parse_markdown_to_json(text)

    def _parse_markdown_to_json(self, text: str) -> Optional[Dict[str, Any]]:
        lines = [line.strip() for line in text.splitlines() if line.strip() and "|" in line]
        if len(lines) < 2:
            return None

        headers = [c.strip() for c in lines[0].strip("|").split("|")]
        data_rows = []
        for line in lines[1:]:
            if re.match(r"^\|?\s*:?-+:?\s*(\|?\s*:?-+:?\s*)*\|?$", line):
                continue
            cols = [c.strip() for c in line.strip("|").split("|")]
            if len(cols) == len(headers):
                data_rows.append(cols)

        if headers and data_rows:
            return {"columns": headers, "rows": data_rows}
        return None

    def _execute_sql_query(
        self,
        query: str,
        table_data: Dict[str, Any],
        table_name: str = "doc_table",
        connection: Optional[Any] = None,
        column_types: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """Generate one SQL statement for *query* and execute it.

        Two execution modes.  With *connection* the table already exists - registered
        at ingestion by :class:`DuckDBTableStore` with types decided once - and this
        only runs the SELECT.  Without it, an in-memory table is created and populated
        per call, which is what happened for every query before the store existed.
        """
        columns = table_data.get("columns", [])
        rows = table_data.get("rows", [])
        if not columns or not rows:
            return "", [], 0.0

        # Lever A: Infer column SQL types.  Types registered at ingestion win, so the
        # same column cannot type differently from one query to the next.
        if column_types is None:
            column_types = {}
            for col_idx, col in enumerate(columns):
                col_values = [row[col_idx] for row in rows if col_idx < len(row)]
                column_types[col] = infer_sql_type(col_values)
        table_data["inferred_types"] = column_types

        safe_table = quote_identifier(table_name)
        is_agg = any(agg in query for agg in ["sum", "average", "avg", "max", "min", "total sum"])

        # Check for year match in columns
        year_match = next(
            (
                c
                for c in columns
                if any(y in c for y in ["2019", "2018", "2020", "2021", "2022", "2023", "2024", "2025"])
            ),
            None,
        )
        col_0 = columns[0] if columns else "col_1"

        # Check for target concept terms in query
        target_concept = None
        for concept in [
            "total assets",
            "total equity",
            "total liabilities",
            "cash and cash equivalents",
            "net income",
            "operating expenses",
        ]:
            if concept in query:
                target_concept = concept
                break

        numeric_cols = [c for c, t in column_types.items() if t in ("DECIMAL(18,2)", "INTEGER")]

        if target_concept:
            concept_keyword = target_concept.split()[0]
            if year_match:
                generated_sql = (
                    f"SELECT {quote_identifier(col_0)}, {quote_identifier(year_match)} FROM {safe_table} "
                    f"WHERE LOWER({quote_identifier(col_0)}) LIKE '%{concept_keyword}%'"
                )
            else:
                generated_sql = (
                    f"SELECT * FROM {safe_table} WHERE LOWER({quote_identifier(col_0)}) LIKE '%{concept_keyword}%'"
                )
        elif is_agg and numeric_cols:
            generated_sql = f"SELECT SUM({quote_identifier(numeric_cols[0])}) AS total_sum FROM {safe_table}"
        elif year_match:
            generated_sql = f"SELECT {quote_identifier(col_0)}, {quote_identifier(year_match)} FROM {safe_table}"
        else:
            generated_sql = f"SELECT * FROM {safe_table}"

        # Execute the single generated statement in an in-memory DuckDB session.
        # NOTE: there is no self-consistency / majority-vote step here — one
        # template produces one statement and it is executed once.
        if connection is not None:
            try:
                # The prepared store is read-only, so a SELECT is all that can run here.
                rel = connection.execute(generated_sql)
                fetched_rows = rel.fetchall()
                col_names = [desc[0] for desc in rel.description]
                result_dicts = [dict(zip(col_names, r)) for r in fetched_rows]

                col_matches = sum(1 for col in columns if str(col).lower() in query)
                score = 0.6 + (0.3 if col_matches > 0 else 0.0)
                return generated_sql, result_dicts, min(1.0, score)
            except Exception as exc:
                logger.warning("Prepared-table execution error: %s - logging failure", exc)
                log_sql_failure(query, generated_sql, str(exc), table_data)

        elif self._has_duckdb():
            try:
                import duckdb

                con = duckdb.connect(database=":memory:")
                col_defs = ", ".join(f"{quote_identifier(col)} {column_types.get(col, 'VARCHAR')}" for col in columns)
                con.execute(f"CREATE TABLE {safe_table} ({col_defs})")

                for row in rows:
                    placeholders = ", ".join(["?"] * len(columns))
                    # Pad row if needed and sanitize typed values (empty strings -> None / NULL)
                    row_padded = row + [None] * (len(columns) - len(row))
                    sanitized_row = [
                        sanitize_cell_for_sql(val, column_types.get(col, "VARCHAR"))
                        for col, val in zip(columns, row_padded[: len(columns)])
                    ]
                    con.execute(f"INSERT INTO {safe_table} VALUES ({placeholders})", sanitized_row)

                rel = con.execute(generated_sql)
                fetched_rows = rel.fetchall()
                col_names = [desc[0] for desc in rel.description]
                result_dicts = [dict(zip(col_names, r)) for r in fetched_rows]
                con.close()

                col_matches = sum(1 for col in columns if str(col).lower() in query)
                score = 0.6 + (0.3 if col_matches > 0 else 0.0)
                return generated_sql, result_dicts, min(1.0, score)
            except Exception as exc:
                logger.warning("DuckDB execution error: %s - logging failure", exc)
                log_sql_failure(query, generated_sql, str(exc), table_data)

        # In-memory Python runner fallback
        result_dicts = [dict(zip(columns, row)) for row in rows[:5]]
        col_matches = sum(1 for col in columns if col.lower() in query)
        score = 0.6 if col_matches or "revenue" in query or "region" in query else 0.2
        return generated_sql, result_dicts, score

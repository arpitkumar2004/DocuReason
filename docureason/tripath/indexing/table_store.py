"""table_store.py — DuckDB tables registered once at ingestion, not per query.

``TableSQLRetriever`` rebuilt every table it touched on **every query**: a
``CREATE TABLE`` plus one parameter-bound ``INSERT`` per row, into a fresh
``:memory:`` database, discarded when the query finished.  For a filing with 40
tables that is 40 schema creations and every row re-inserted per question asked, and
the inferred column types were re-derived each time - so the same table could type a
column as ``VARCHAR`` for one query and ``DECIMAL`` for another depending on which
rows the parser happened to produce.

Registering at ingestion fixes both: the schema is decided once, recorded in
``tables_index.json`` next to the corpus it describes, and queries run against a
prepared read-only database.

Layout, inside the run's output directory::

    tables.duckdb        one table per table-region
    tables_index.json    region_id -> {table_name, columns, column_types, row_count}

The index is the contract.  A region absent from it has no prepared table, and the
retriever falls back to the in-memory path rather than guessing a table name - so a
partially built store degrades instead of raising.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from docureason.tripath.utils import get_logger

logger = get_logger(__name__)

DB_FILENAME = "tables.duckdb"
INDEX_FILENAME = "tables_index.json"

#: DuckDB identifiers are quoted everywhere in this module, but a table name derived
#: from a region id is also used as a dict key and logged, so it is normalised to a
#: conservative character set rather than quoted-and-hoped.
_UNSAFE = re.compile(r"[^0-9a-zA-Z_]+")


def table_name_for(region_id: str) -> str:
    """Deterministic SQL table name for *region_id*.

    Deterministic matters: the name is stored in ``tables_index.json`` at ingestion
    and looked up at query time, in a different process.
    """
    slug = _UNSAFE.sub("_", str(region_id)).strip("_").lower()
    if not slug:
        slug = "unnamed"
    if slug[0].isdigit():
        slug = f"t_{slug}"
    return slug[:180]


@dataclass
class RegisteredTable:
    """One table-region's prepared schema."""

    region_id: str
    document_id: str
    table_name: str
    columns: List[str]
    column_types: Dict[str, str]
    row_count: int
    page_no: Optional[int] = None
    unit_multiplier: int = 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "document_id": self.document_id,
            "table_name": self.table_name,
            "columns": list(self.columns),
            "column_types": dict(self.column_types),
            "row_count": self.row_count,
            "page_no": self.page_no,
            "unit_multiplier": self.unit_multiplier,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RegisteredTable":
        return cls(
            region_id=str(payload.get("region_id", "")),
            document_id=str(payload.get("document_id", "")),
            table_name=str(payload.get("table_name", "")),
            columns=list(payload.get("columns") or []),
            column_types=dict(payload.get("column_types") or {}),
            row_count=int(payload.get("row_count") or 0),
            page_no=payload.get("page_no"),
            unit_multiplier=int(payload.get("unit_multiplier") or 1),
        )


@dataclass
class TableStoreStats:
    """What a build actually registered, for the ingestion audit."""

    tables_registered: int = 0
    rows_registered: int = 0
    regions_skipped: int = 0
    degraded: Optional[str] = None
    skipped_reasons: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tables_registered": self.tables_registered,
            "rows_registered": self.rows_registered,
            "regions_skipped": self.regions_skipped,
            "skipped_reasons": dict(self.skipped_reasons),
            "degraded": self.degraded,
        }


class DuckDBTableStore:
    """Build and read the per-run DuckDB table store."""

    def __init__(self, output_dir: Union[str, Path]) -> None:
        self.output_dir = Path(output_dir)
        self.db_path = self.output_dir / DB_FILENAME
        self.index_path = self.output_dir / INDEX_FILENAME
        self._index: Optional[Dict[str, RegisteredTable]] = None
        self._connection: Any = None

    # -- availability -------------------------------------------------------

    @staticmethod
    def duckdb_available() -> bool:
        try:
            import duckdb  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        """True when a prepared store exists on disk and DuckDB can read it."""
        return self.db_path.exists() and self.index_path.exists() and self.duckdb_available()

    # -- build (ingestion time) --------------------------------------------

    def build(self, documents: Sequence[Any]) -> TableStoreStats:
        """Register every table-region in *documents* as a DuckDB table.

        Rebuilds from scratch: a stale table left over from a previous corpus in the
        same directory would still be reachable by name and would answer with another
        document's numbers, which is worse than having no store at all.
        """
        stats = TableStoreStats()
        if not self.duckdb_available():
            stats.degraded = "duckdb_unavailable"
            logger.warning("DuckDBTableStore: duckdb not installed - tables will be built per query")
            return stats

        import duckdb

        from ..retrieval.table_sql import infer_sql_type, quote_identifier, sanitize_cell_for_sql

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.unlink(missing_ok=True)
        index: Dict[str, RegisteredTable] = {}

        try:
            con = duckdb.connect(str(self.db_path))
        except Exception as exc:
            stats.degraded = "connect_failed"
            logger.error("DuckDBTableStore: cannot open %s: %s", self.db_path, exc)
            return stats

        try:
            for document in documents:
                document_id = str(getattr(document, "id", "") or "")
                for region in getattr(document, "regions", []) or []:
                    if getattr(region, "type", "") != "table":
                        continue
                    metadata = getattr(region, "metadata", {}) or {}
                    region_id = str(metadata.get("region_id") or "")
                    payload = _table_payload(region)

                    reason = _rejection_reason(region_id, payload)
                    if reason:
                        stats.regions_skipped += 1
                        stats.skipped_reasons[reason] = stats.skipped_reasons.get(reason, 0) + 1
                        continue

                    columns = [str(c) for c in payload["columns"]]
                    rows = payload["rows"]
                    name = _unique_name(table_name_for(region_id), index)

                    column_types: Dict[str, str] = {}
                    for position, column in enumerate(columns):
                        values = [row[position] for row in rows if position < len(row)]
                        column_types[column] = infer_sql_type(values)

                    col_defs = ", ".join(f"{quote_identifier(c)} {column_types[c]}" for c in columns)
                    con.execute(f"CREATE TABLE {quote_identifier(name)} ({col_defs})")
                    placeholders = ", ".join(["?"] * len(columns))
                    for row in rows:
                        padded = list(row) + [None] * (len(columns) - len(row))
                        con.execute(
                            f"INSERT INTO {quote_identifier(name)} VALUES ({placeholders})",
                            [
                                sanitize_cell_for_sql(value, column_types[column])
                                for column, value in zip(columns, padded[: len(columns)])
                            ],
                        )

                    index[region_id] = RegisteredTable(
                        region_id=region_id,
                        document_id=document_id,
                        table_name=name,
                        columns=columns,
                        column_types=column_types,
                        row_count=len(rows),
                        page_no=metadata.get("page_no"),
                        unit_multiplier=int(payload.get("unit_multiplier") or 1),
                    )
                    stats.tables_registered += 1
                    stats.rows_registered += len(rows)
        except Exception as exc:
            stats.degraded = "build_error"
            logger.error("DuckDBTableStore: build failed: %s", exc)
        finally:
            con.close()

        self.index_path.write_text(
            json.dumps({rid: entry.as_dict() for rid, entry in index.items()}, indent=2),
            encoding="utf-8",
        )
        self._index = index
        logger.info(
            "DuckDBTableStore: registered %d table(s) / %d row(s) in %s (skipped %d)",
            stats.tables_registered,
            stats.rows_registered,
            self.db_path.name,
            stats.regions_skipped,
        )
        return stats

    # -- read (query time) --------------------------------------------------

    def index(self) -> Dict[str, RegisteredTable]:
        """The region_id to prepared-table map, read from disk once."""
        if self._index is None:
            if not self.index_path.exists():
                self._index = {}
            else:
                try:
                    raw = json.loads(self.index_path.read_text(encoding="utf-8"))
                    self._index = {rid: RegisteredTable.from_dict(entry) for rid, entry in raw.items()}
                except Exception as exc:
                    logger.warning("DuckDBTableStore: unreadable index %s: %s", self.index_path, exc)
                    self._index = {}
        return self._index

    def lookup(self, region_id: str) -> Optional[RegisteredTable]:
        return self.index().get(str(region_id))

    def connection(self) -> Any:
        """A cached read-only connection, or None when there is nothing to read.

        Read-only is deliberate: a query path able to write to the store could mutate
        the corpus it is supposed to be reporting on.
        """
        if self._connection is not None:
            return self._connection
        if not self.available:
            return None
        try:
            import duckdb

            self._connection = duckdb.connect(str(self.db_path), read_only=True)
        except Exception as exc:
            logger.warning("DuckDBTableStore: cannot open %s read-only: %s", self.db_path, exc)
            self._connection = None
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None


def _table_payload(region: Any) -> Optional[Dict[str, Any]]:
    """The columns/rows payload for *region*, from the object or from its metadata."""
    raw = getattr(region, "table_json", None)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = None
    if not isinstance(raw, dict):
        metadata = getattr(region, "metadata", {}) or {}
        candidate = metadata.get("table_json")
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except Exception:
                candidate = None
        raw = candidate if isinstance(candidate, dict) else None
    return raw


def _rejection_reason(region_id: str, payload: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why this region cannot be registered, or None when it can."""
    if not region_id:
        return "no_region_id"
    if not payload:
        return "no_table_json"
    columns = payload.get("columns") or []
    rows = payload.get("rows") or []
    if not columns:
        return "no_columns"
    if not rows:
        return "no_rows"
    if len({str(c) for c in columns}) != len(columns):
        # Duplicate headers cannot become distinct SQL columns. The in-memory path has
        # the same problem, but it fails per query rather than poisoning the store.
        return "duplicate_columns"
    return None


def _unique_name(base: str, index: Dict[str, RegisteredTable]) -> str:
    """*base*, suffixed when another region already claimed that name."""
    taken = {entry.table_name for entry in index.values()}
    if base not in taken:
        return base
    suffix = 2
    while f"{base}_{suffix}" in taken:
        suffix += 1
    return f"{base}_{suffix}"


__all__ = [
    "DB_FILENAME",
    "INDEX_FILENAME",
    "DuckDBTableStore",
    "RegisteredTable",
    "TableStoreStats",
    "table_name_for",
]

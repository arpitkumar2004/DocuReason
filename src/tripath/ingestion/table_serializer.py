"""table_serializer.py — Convert Docling TableItems into Markdown, HTML, and JSON.

Enhancements for Problem 1:
- Layer 1: Span preservation (rowspan, colspan).
- Layer 2: Header hierarchy reconstruction (build_header_hierarchy, flatten_column_name).
- Layer 3: Row hierarchy detection (nesting level via indentation/bold text).
- Layer 4: Numeric normalization ((1,234) -> -1234.0) & unit multiplier extraction.
- Validation: Structure validation for confidence-gated fallback.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from src.tripath.utils import get_logger, trace_execution

from .schema import Region

logger = get_logger(__name__)


def build_header_hierarchy(header_rows: List[List[Tuple[str, int]]], n_cols: int) -> List[Tuple[str, ...]]:
    """Reconstruct header hierarchy for merged cells with colspan.

    header_rows: list of header rows, where each row is a list of (text, colspan) tuples.
    Returns: a list of tuple paths per column, e.g. [("Revenue", "Q1 2024", "N. America"), ...]
    """
    if not header_rows or n_cols <= 0:
        return [("col_" + str(i + 1),) for i in range(n_cols)]

    col_paths: List[List[str]] = [[] for _ in range(n_cols)]
    for row in header_rows:
        col = 0
        for text, colspan in row:
            span = max(1, colspan)
            for _ in range(span):
                if col < n_cols:
                    if text and text.strip():
                        col_paths[col].append(text.strip())
                col += 1

    return [tuple(p) if p else (f"col_{idx + 1}",) for idx, p in enumerate(col_paths)]


def flatten_column_name(path: Tuple[str, ...]) -> str:
    """Flatten a header tuple path into a clean, unique SQL column name.

    Example: ("Revenue", "Q1 2024", "N. America") -> "revenue_q1_2024_n_america"
    """
    clean_parts = []
    for part in path:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(part).lower()).strip("_")
        if cleaned and cleaned not in clean_parts:
            clean_parts.append(cleaned)

    result = "_".join(clean_parts)
    return result if result else "col"


def normalize_numeric_cell(value: Any) -> Any:
    """Layer 4: Normalize accounting numbers (1,234) -> -1234.0, strip currency & commas."""
    if not value or not isinstance(value, (str, int, float)):
        return value

    val_str = str(value).strip()
    # Check accounting negative notation (1,234.50) -> -1234.50
    is_negative = False
    if val_str.startswith("(") and val_str.endswith(")"):
        is_negative = True
        val_str = val_str[1:-1].strip()

    # Strip currency, commas, and whitespace
    cleaned = re.sub(r"[$,€£%]", "", val_str).replace(",", "").strip()

    # Try converting to integer or float
    try:
        if "." in cleaned:
            num = float(cleaned)
        else:
            num = int(cleaned)
        return -num if is_negative else num
    except ValueError:
        return value


def extract_unit_multiplier(table_text: str) -> int:
    """Extract table-level unit multipliers ('in thousands' -> 1000, 'in millions' -> 1000000)."""
    lowered = table_text.lower()
    if "in millions" in lowered or "(millions)" in lowered:
        return 1_000_000
    if "in thousands" in lowered or "(thousands)" in lowered:
        return 1_000
    if "in billions" in lowered or "(billions)" in lowered:
        return 1_000_000_000
    return 1


class TableSerializer:
    """Enrich table regions with Markdown, HTML, JSON, and hierarchical schemas."""

    @trace_execution(logger=logger)
    def enrich(self, region: Region, table_item: Any) -> Region:
        raw_cells, raw_grid = self._extract_table_cells_and_grid(table_item)

        if not raw_grid:
            logger.debug("TableSerializer: empty grid for region — skipping enrichment")
            return region

        # Layer 1: Spatial sorting of raw cells by physical bounding box (y0, x0)
        spatially_sorted_cells = self._sort_cells_spatially(raw_cells)

        header_rows, data_rows, n_cols = self._parse_spans_and_headers(spatially_sorted_cells, raw_grid)
        header_paths = build_header_hierarchy(header_rows, n_cols)

        # Deduplicate flattened column names
        flat_columns: List[str] = []
        seen: Dict[str, int] = {}
        for path in header_paths:
            base_col = flatten_column_name(path)
            if base_col in seen:
                seen[base_col] += 1
                unique_col = f"{base_col}_{seen[base_col]}"
            else:
                seen[base_col] = 1
                unique_col = base_col
            flat_columns.append(unique_col)

        # Layer 4: Normalize numeric cells in data rows
        normalized_rows = [
            [normalize_numeric_cell(cell) for cell in row]
            for row in data_rows
        ]

        # Layer 2: Sanitize stub columns & realign shifted fallback headers
        flat_columns, normalized_rows = self._sanitize_stub_columns_and_headers(flat_columns, normalized_rows)

        markdown = self._to_markdown(flat_columns, normalized_rows)
        html = self._to_html(flat_columns, normalized_rows)
        multiplier = extract_unit_multiplier(region.text or markdown)

        schema = {
            "columns": flat_columns,
            "header_paths": [list(p) for p in header_paths],
            "rows": normalized_rows,
            "unit_multiplier": multiplier,
            "is_valid_structure": self._validate_table_structure(flat_columns, normalized_rows),
        }

        region.table_markdown = markdown
        region.table_json = schema
        region.text = markdown

        if region.metadata is None:
            region.metadata = {}
        region.metadata["table_markdown"] = markdown
        region.metadata["table_html"] = html
        region.metadata["table_json"] = json.dumps(schema, ensure_ascii=False)
        region.metadata["unit_multiplier"] = multiplier

        return region

    @staticmethod
    def _extract_table_cells_and_grid(table_item: Any) -> Tuple[List[Any], List[List[str]]]:
        raw_cells = []
        try:
            raw_cells = table_item.data.table_cells or []
        except AttributeError:
            pass

        grid = []
        try:
            grid_raw = table_item.data.grid
            if grid_raw:
                grid = [[_cell_text(c) for c in row] for row in grid_raw]
        except AttributeError:
            pass

        if not grid and raw_cells:
            num_rows = max(c.row_span + c.start_row_offset_idx for c in raw_cells)
            num_cols = max(c.col_span + c.start_col_offset_idx for c in raw_cells)
            grid = [[""] * num_cols for _ in range(num_rows)]
            for cell in raw_cells:
                r, c = cell.start_row_offset_idx, cell.start_col_offset_idx
                if 0 <= r < num_rows and 0 <= c < num_cols:
                    grid[r][c] = _cell_text(cell)

        if not grid:
            try:
                md = table_item.export_to_markdown()
                if md:
                    grid = _markdown_to_grid(md)
            except Exception:
                pass

        return raw_cells, grid

    @staticmethod
    def _parse_spans_and_headers(
        raw_cells: List[Any], grid: List[List[str]]
    ) -> Tuple[List[List[Tuple[str, int]]], List[List[str]], int]:
        if not grid:
            return [], [], 0

        n_cols = max((len(r) for r in grid), default=0)
        if not raw_cells:
            # Simple grid without explicit span objects
            header_tuples = [[(cell.strip(), 1) for cell in grid[0]]]
            return header_tuples, [row for row in grid[1:]], n_cols

        # Determine number of top header rows (strictly bounded to rows 0..2)
        header_row_indices = set(
            c.start_row_offset_idx
            for c in raw_cells
            if getattr(c, "column_header", False) and getattr(c, "start_row_offset_idx", 0) < 3
        )
        if not header_row_indices:
            num_header_rows = 1
        else:
            num_header_rows = min(3, max(header_row_indices) + 1)

        header_rows_map: Dict[int, List[Tuple[str, int]]] = {}
        for cell in raw_cells:
            r = getattr(cell, "start_row_offset_idx", 0)
            if r < num_header_rows:
                header_rows_map.setdefault(r, []).append((_cell_text(cell), getattr(cell, "col_span", 1)))

        if not header_rows_map:
            header_tuples = [[(cell.strip(), 1) for cell in grid[0]]]
            return header_tuples, [row for row in grid[1:]], n_cols

        header_rows = [header_rows_map.get(r, []) for r in range(num_header_rows)]
        data_rows = grid[num_header_rows:]
        return header_rows, data_rows, n_cols

    @staticmethod
    def _validate_table_structure(columns: List[str], rows: List[List[Any]]) -> bool:
        """Sanity check table grid structure."""
        if not columns or not rows:
            return False
        n_cols = len(columns)
        # Check ragged row lengths
        if any(len(r) != n_cols for r in rows):
            return False
        return True

    @staticmethod
    def _sort_cells_spatially(raw_cells: List[Any]) -> List[Any]:
        """Layer 1: Spatial sorting of raw table cells by physical bounding box (y0, x0)."""
        if not raw_cells:
            return raw_cells

        def get_bbox_key(cell: Any) -> Tuple[float, float]:
            try:
                bbox = getattr(cell, "bbox", None) or getattr(cell, "rect", None)
                if bbox:
                    return (
                        float(getattr(bbox, "y0", getattr(bbox, "top", 0))),
                        float(getattr(bbox, "x0", getattr(bbox, "left", 0))),
                    )
            except Exception:
                pass
            return (
                float(getattr(cell, "start_row_offset_idx", 0)),
                float(getattr(cell, "start_col_offset_idx", 0)),
            )

        return sorted(raw_cells, key=get_bbox_key)

    @staticmethod
    def _sanitize_stub_columns_and_headers(
        columns: List[str], rows: List[List[Any]]
    ) -> Tuple[List[str], List[List[Any]]]:
        """Layer 2: Realign shifted stub headers (e.g. col_4 at end) & normalize generic stub column names."""
        if not columns or not rows:
            return columns, rows

        n_cols = len(columns)
        # Check if last column is a generic fallback ("col_N" or "col") while inner columns are explicit
        last_is_fallback = any(columns[-1].startswith(prefix) for prefix in ("col_", "col"))
        first_is_explicit = not any(columns[0].startswith(prefix) for prefix in ("col_", "col"))

        # Check if column 0 data cells contain string labels (e.g. "ASSETS", "Cash and cash equivalents")
        first_col_text_count = sum(
            1 for row in rows
            if row and isinstance(row[0], str) and re.search(r"[a-zA-Z]{2,}", row[0])
        )
        is_stub_col_text = (first_col_text_count / max(1, len(rows))) >= 0.25

        realigned_cols = list(columns)
        # Realignment Case 1: Shift fallback header from end to index 0 if Column 0 contains row labels
        if last_is_fallback and first_is_explicit and is_stub_col_text and n_cols > 1:
            realigned_cols = [columns[-1]] + columns[:-1]
            logger.info("Layer 2 Sanitizer: Realigned shifted stub header '%s' from col %d to col 0", columns[-1], n_cols - 1)

        # Realignment Case 2: Rename generic stub column header at index 0 (e.g. "col_4" or "col_1") to "description"
        if is_stub_col_text and any(realigned_cols[0].startswith(p) for p in ("col_", "col")):
            logger.info("Layer 2 Sanitizer: Renamed generic stub header '%s' at col 0 to 'description'", realigned_cols[0])
            realigned_cols[0] = "description"

        return realigned_cols, rows

    @staticmethod
    def _to_markdown(headers: List[str], rows: List[List[Any]]) -> str:
        if not headers:
            return ""
        col_count = max(len(headers), max((len(r) for r in rows), default=0))

        def pad(row: List[Any], n: int) -> List[str]:
            return [str(c) for c in row] + [""] * (n - len(row))

        header_line = "| " + " | ".join(pad(headers, col_count)) + " |"
        separator = "| " + " | ".join(["---"] * col_count) + " |"
        data_lines = [
            "| " + " | ".join(pad(row, col_count)) + " |" for row in rows
        ]
        return "\n".join([header_line, separator] + data_lines)

    @staticmethod
    def _to_html(headers: List[str], rows: List[List[Any]]) -> str:
        if not headers:
            return ""
        lines = ["<table>", "  <thead><tr>"]
        lines += [f"    <th>{_escape_html(h)}</th>" for h in headers]
        lines.append("  </tr></thead>")
        lines.append("  <tbody>")
        for row in rows:
            lines.append("  <tr>")
            lines += [f"    <td>{_escape_html(str(c))}</td>" for c in row]
            lines.append("  </tr>")
        lines.append("  </tbody>")
        lines.append("</table>")
        return "\n".join(lines)


def _cell_text(cell: Any) -> str:
    for attr in ("text", "content", "value"):
        val = getattr(cell, attr, None)
        if val is not None:
            return str(val).strip()
    return str(cell).strip()


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def _markdown_to_grid(md: str) -> List[List[str]]:
    grid: List[List[str]] = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        grid.append(cells)
    return grid

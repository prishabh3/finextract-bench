"""
FinExtract-Bench: PyMuPDF-based text parser.

Extracts plain text, text blocks with bounding boxes, and tables from PDFs.
Uses PyMuPDF's native table detection (page.find_tables()) — no ML required.

Coordinate system: PyMuPDF normalises all coordinates to top-left origin
(0, 0) = top-left of page. Bboxes are (x0, y0, x1, y1).

This parser is used for Pipeline A (text-only) and as a lightweight fallback
when Docling is not installed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import pymupdf  # PyMuPDF >= 1.24; preferred import name
except ImportError:
    try:
        import fitz as pymupdf  # Legacy alias — acceptable fallback
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for text parsing. "
            "Install it with: pip install pymupdf"
        ) from exc

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

from finextract.parsing.base import (
    ParsedCell,
    ParsedDocument,
    ParsedPage,
    ParsedTable,
    TextBlock,
)

# Version string for provenance
_PYMUPDF_VERSION = getattr(pymupdf, "version", ("unknown",))[0]
PARSER_NAME = "pymupdf"
PARSER_VERSION = _PYMUPDF_VERSION


def parse_pdf(
    file_path: Path,
    document_id: str,
    *,
    extract_tables: bool = True,
    max_pages: int | None = None,
) -> ParsedDocument:
    """
    Parse a PDF with PyMuPDF and return a ParsedDocument.

    Args:
        file_path: Absolute path to the PDF file.
        document_id: Document UUID assigned at ingestion.
        extract_tables: Whether to run table detection (page.find_tables()).
        max_pages: Limit the number of pages parsed (None = all pages).

    Returns:
        ParsedDocument with all pages, text blocks, and tables.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If the file cannot be opened as a PDF.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    t_start = time.monotonic()
    pages: list[ParsedPage] = []

    with pymupdf.open(str(file_path)) as doc:
        total_pages = len(doc)
        pages_to_parse = (
            min(max_pages, total_pages) if max_pages else total_pages
        )

        for page_idx in range(pages_to_parse):
            page = doc[page_idx]
            page_number = page_idx + 1  # 1-indexed
            parsed_page = _parse_page(page, page_number, extract_tables=extract_tables)
            pages.append(parsed_page)

    elapsed_ms = (time.monotonic() - t_start) * 1000

    parsed_doc = ParsedDocument(
        document_id=document_id,
        filename=file_path.name,
        file_path=file_path,
        page_count=total_pages,
        pages=pages,
        parser_name=PARSER_NAME,
        parser_version=str(PARSER_VERSION),
        parse_time_ms=elapsed_ms,
    )

    logger.info(
        "PyMuPDF parsed %s: %d pages, %d tables, %.0f ms",
        file_path.name,
        total_pages,
        sum(len(p.tables) for p in pages),
        elapsed_ms,
    )
    return parsed_doc


def _parse_page(page, page_number: int, *, extract_tables: bool) -> ParsedPage:
    """
    Extract all content from a single PyMuPDF page object.

    Returns a ParsedPage with full_text, text_blocks, and tables.
    """
    width = page.rect.width
    height = page.rect.height

    # ---- Full plain text ----
    full_text: str = page.get_text("text") or ""

    # ---- Text blocks with bounding boxes ----
    # get_text("blocks") returns:
    #   (x0, y0, x1, y1, text, block_no, block_type)
    # block_type == 0 → text block; block_type == 1 → image
    raw_blocks = page.get_text("blocks") or []
    text_blocks: list[TextBlock] = []

    for block_no, block in enumerate(raw_blocks):
        x0, y0, x1, y1, text, _, block_type = block
        if block_type != 0:
            continue  # Skip image blocks
        stripped = text.strip()
        if not stripped:
            continue
        text_blocks.append(
            TextBlock(
                text=stripped,
                page=page_number,
                bbox=[x0, y0, x1, y1],
                block_index=block_no,
            )
        )

    # ---- Tables ----
    tables: list[ParsedTable] = []
    if extract_tables:
        tables = _extract_tables(page, page_number)

    return ParsedPage(
        page_number=page_number,
        full_text=full_text,
        text_blocks=text_blocks,
        tables=tables,
        width=width,
        height=height,
    )


def _extract_tables(page, page_number: int) -> list[ParsedTable]:
    """
    Detect and extract tables from a single page using page.find_tables().

    Returns a list of ParsedTable instances.
    """
    tables: list[ParsedTable] = []

    try:
        finder = page.find_tables()
    except Exception as exc:
        logger.warning("Table detection failed on page %d: %s", page_number, exc)
        return tables

    for table_idx, table in enumerate(finder.tables):
        try:
            rows = table.extract()  # list[list[str | None]]
            if not rows:
                continue

            # Convert None cells to empty strings
            cleaned_rows = [
                [cell if cell is not None else "" for cell in row]
                for row in rows
            ]

            # Detect header: first row is used as header if it looks non-numeric
            headers: list[str] = []
            if cleaned_rows:
                first_row = cleaned_rows[0]
                if _looks_like_header(first_row):
                    headers = first_row
                    data_rows = cleaned_rows[1:]
                else:
                    data_rows = cleaned_rows
            else:
                data_rows = []

            # Build cell list (with table bbox as fallback for individual cells)
            table_bbox = _rect_to_list(table.bbox)
            cells: list[ParsedCell] = []
            for r_idx, row in enumerate(cleaned_rows):
                for c_idx, cell_text in enumerate(row):
                    cells.append(ParsedCell(
                        text=cell_text,
                        row=r_idx,
                        col=c_idx,
                        bbox=None,  # cell-level bboxes not extracted here
                    ))

            # Build DataFrame if pandas available
            dataframe = None
            if _PANDAS_AVAILABLE:
                try:
                    dataframe = table.to_pandas()
                except Exception:
                    if data_rows and headers:
                        import pandas as pd
                        dataframe = pd.DataFrame(data_rows, columns=headers)

            parsed_table = ParsedTable(
                page=page_number,
                bbox=table_bbox,
                rows=cleaned_rows,
                headers=headers,
                dataframe=dataframe,
                cells=cells,
                table_index=table_idx,
            )
            tables.append(parsed_table)

        except Exception as exc:
            logger.warning(
                "Failed to extract table %d on page %d: %s",
                table_idx, page_number, exc,
            )

    return tables


def _rect_to_list(rect) -> list[float] | None:
    """Convert a PyMuPDF Rect to [x0, y0, x1, y1] list, or None."""
    try:
        return [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
    except Exception:
        return None


def _looks_like_header(row: list[str]) -> bool:
    """
    Heuristic: a row looks like a header if it contains mostly non-numeric text.

    Returns True if fewer than half the non-empty cells are numeric.
    """
    non_empty = [cell for cell in row if cell.strip()]
    if not non_empty:
        return False
    numeric_count = sum(1 for cell in non_empty if _is_numeric_cell(cell))
    return numeric_count < len(non_empty) / 2


def _is_numeric_cell(text: str) -> bool:
    """Return True if cell text looks like a financial number."""
    import re
    cleaned = re.sub(r"[\$€£¥,\s%()]", "", text.strip())
    if cleaned.startswith("-"):
        cleaned = cleaned[1:]
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def extract_text_near_keyword(
    parsed_doc: ParsedDocument,
    keyword: str,
    *,
    context_chars: int = 500,
    case_sensitive: bool = False,
) -> list[dict]:
    """
    Find all occurrences of a keyword and return surrounding text context.

    Used to locate financial section headers like 'Total net sales' or
    'Revenue' before passing context to an LLM extractor.

    Returns:
        List of dicts with keys: page, position, context, bbox.
    """
    results = []
    search_kw = keyword if case_sensitive else keyword.lower()

    for page in parsed_doc.pages:
        text = page.full_text if case_sensitive else page.full_text.lower()
        pos = 0
        while True:
            idx = text.find(search_kw, pos)
            if idx == -1:
                break
            start = max(0, idx - context_chars // 2)
            end = min(len(page.full_text), idx + len(keyword) + context_chars // 2)
            results.append({
                "page": page.page_number,
                "position": idx,
                "context": page.full_text[start:end],
                "bbox": None,  # Not available from full_text scan
            })
            pos = idx + 1

    return results

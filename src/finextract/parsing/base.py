"""
FinExtract-Bench: Core data structures for parsed PDF documents.

These dataclasses form the contract between parsers and the extraction layer.
All parsers (text-only, Docling) must return a ParsedDocument.

Design decisions:
- Immutable dataclasses (frozen where possible) to prevent accidental mutation.
- All provenance fields are Optional — a text-only parser has no bboxes.
- Tables carry both a raw DataFrame and the original string rows so extraction
  can work from either representation.
- Page numbers are always 1-indexed throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ============================================================
# Fine-grained parsed elements
# ============================================================


@dataclass
class TextBlock:
    """
    A contiguous block of text from one page.

    For PyMuPDF text-only parsing, bbox is always present.
    For Docling, bbox comes from item.prov[0].bbox.
    """

    text: str
    page: int  # 1-indexed
    bbox: list[float] | None = None  # [x0, y0, x1, y1] top-left origin
    block_index: int | None = None  # Order within the page
    font_size: float | None = None
    is_bold: bool = False


@dataclass
class ParsedCell:
    """One cell within a parsed table."""

    text: str
    row: int
    col: int
    bbox: list[float] | None = None


@dataclass
class ParsedTable:
    """
    A financial table detected on a page.

    `rows` contains the raw string grid (list of lists).
    `dataframe` is a pandas DataFrame representation (may be None if pandas
    is not available, though it is a required dependency).
    `headers` is the first row of the table when it looks like a header row.
    """

    page: int  # 1-indexed
    bbox: list[float] | None = None  # [x0, y0, x1, y1]
    rows: list[list[str]] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    dataframe: Any = None  # pandas DataFrame or None
    cells: list[ParsedCell] = field(default_factory=list)
    table_index: int = 0  # Order within the page
    caption: str | None = None


@dataclass
class ParsedPage:
    """
    All extracted content from one PDF page.

    `full_text` is the complete plain-text of the page (for quick scanning).
    `text_blocks` are the individual paragraph/line blocks.
    `tables` are the detected financial tables.
    """

    page_number: int  # 1-indexed
    full_text: str = ""
    text_blocks: list[TextBlock] = field(default_factory=list)
    tables: list[ParsedTable] = field(default_factory=list)
    width: float | None = None   # Page width in points
    height: float | None = None  # Page height in points


# ============================================================
# Top-level parsed document
# ============================================================


@dataclass
class ParsedDocument:
    """
    The complete parsed representation of one PDF document.

    This is the primary output of any DocumentParser implementation.
    Downstream components (extractors, provenance tracker) consume this.
    """

    document_id: str
    filename: str
    file_path: Path
    page_count: int
    pages: list[ParsedPage] = field(default_factory=list)

    # Parser metadata — for provenance recording
    parser_name: str = "unknown"
    parser_version: str = "unknown"
    parsed_at: datetime = field(default_factory=datetime.utcnow)

    # Timing (milliseconds)
    parse_time_ms: float | None = None

    def all_text(self) -> str:
        """Concatenate full text from all pages (for quick keyword search)."""
        return "\n".join(p.full_text for p in self.pages)

    def all_tables(self) -> list[ParsedTable]:
        """Flatten tables from all pages into a single list."""
        tables = []
        for page in self.pages:
            tables.extend(page.tables)
        return tables

    def get_page(self, page_number: int) -> ParsedPage | None:
        """Return a page by 1-indexed page number, or None."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

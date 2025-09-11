"""
FinExtract-Bench: Docling layout-aware PDF parser.

Uses IBM Docling (v2.x) to produce a structured document representation
with table detection, reading-order preservation, and page-level provenance.

Docling API used (verified against v2.x):
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    result = converter.convert(path)
    doc = result.document          # DoclingDocument
    doc.iterate_items()            # yields (item, level)
    item.prov[0].page_no           # 1-indexed page number
    item.prov[0].bbox              # BoundingBox(l, t, r, b, coord_origin)
    item.export_to_dataframe()     # pandas DataFrame (TableItem only)
    item.text                      # str (TextItem)

Coordinate system:
    Docling PDFs use CoordOrigin.BOTTOMLEFT by default. We convert to
    top-left origin (matching PyMuPDF) using bbox.to_top_left_origin(page_height).

This parser is used for Pipeline B (layout-aware) and C (hybrid).
Requires: pip install docling  (pulls in torch / transformers ~2 GB)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Optional import — clear error if Docling not installed ────────────
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import TableItem, TextItem

    _DOCLING_AVAILABLE = True
    _DOCLING_VERSION = "2.x"  # Will be overridden below
    try:
        import importlib.metadata
        _DOCLING_VERSION = importlib.metadata.version("docling")
    except Exception:
        pass
except ImportError:
    _DOCLING_AVAILABLE = False
    _DOCLING_VERSION = "not installed"

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

from finextract.config.settings import settings
from finextract.parsing.base import (
    ParsedCell,
    ParsedDocument,
    ParsedPage,
    ParsedTable,
    TextBlock,
)

PARSER_NAME = "docling"
PARSER_VERSION = _DOCLING_VERSION


def _require_docling() -> None:
    """Raise ImportError with install instructions if Docling is unavailable."""
    if not _DOCLING_AVAILABLE:
        raise ImportError(
            "Docling is required for layout-aware parsing. "
            "Install it with: pip install docling\n"
            "Note: this pulls in PyTorch and ~2 GB of ML model weights."
        )


def _build_converter() -> "DocumentConverter":
    """
    Build a DocumentConverter with pipeline options from application settings.

    Called once per parse() invocation (Docling models are cached internally
    by Docling between calls in the same process).
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = settings.docling_do_table_structure
    pipeline_options.do_ocr = settings.docling_do_ocr

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def parse_pdf(
    file_path: Path,
    document_id: str,
    *,
    max_pages: int | None = None,
) -> ParsedDocument:
    """
    Parse a PDF with Docling and return a ParsedDocument.

    Args:
        file_path: Absolute path to the PDF file.
        document_id: Document UUID assigned at ingestion.
        max_pages: Limit the number of pages parsed (None = all pages).
            Note: Docling always parses the full document; this filter is
            applied post-parse to trim the returned pages.

    Returns:
        ParsedDocument with structured text blocks and tables including
        page numbers and bounding boxes.

    Raises:
        ImportError: If Docling is not installed.
        FileNotFoundError: If file_path does not exist.
    """
    _require_docling()

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    t_start = time.monotonic()

    converter = _build_converter()
    result = converter.convert(str(file_path))
    doc = result.document  # DoclingDocument

    # ── Collect page dimensions (needed for coordinate conversion) ──
    page_heights: dict[int, float] = {}
    try:
        for page_no, page_obj in doc.pages.items():
            if hasattr(page_obj, "size") and page_obj.size:
                page_heights[page_no] = float(page_obj.size.height)
    except Exception:
        pass  # page_heights will be empty; bbox conversion will skip

    # ── Build page_number → ParsedPage mapping ──
    page_map: dict[int, ParsedPage] = {}

    # Pass 1: collect TextItems
    text_block_idx: dict[int, int] = {}  # page → running block index

    for item, _level in doc.iterate_items():
        if not isinstance(item, TextItem):
            continue
        if not item.prov:
            continue

        prov = item.prov[0]
        page_no: int = prov.page_no  # 1-indexed

        if page_no not in page_map:
            page_map[page_no] = ParsedPage(page_number=page_no)
            text_block_idx[page_no] = 0

        bbox = _convert_bbox(prov.bbox, page_heights.get(page_no))

        block = TextBlock(
            text=item.text,
            page=page_no,
            bbox=bbox,
            block_index=text_block_idx[page_no],
        )
        page_map[page_no].text_blocks.append(block)
        page_map[page_no].full_text += item.text + "\n"
        text_block_idx[page_no] += 1

    # Pass 2: collect TableItems
    table_idx_per_page: dict[int, int] = {}

    for item, _level in doc.iterate_items():
        if not isinstance(item, TableItem):
            continue
        if not item.prov:
            continue

        prov = item.prov[0]
        page_no = prov.page_no

        if page_no not in page_map:
            page_map[page_no] = ParsedPage(page_number=page_no)
        if page_no not in table_idx_per_page:
            table_idx_per_page[page_no] = 0

        bbox = _convert_bbox(prov.bbox, page_heights.get(page_no))
        parsed_table = _convert_table_item(item, page_no, bbox, table_idx_per_page[page_no])
        page_map[page_no].tables.append(parsed_table)
        table_idx_per_page[page_no] += 1

    # ── Assemble pages in order ──
    sorted_pages = [page_map[pn] for pn in sorted(page_map.keys())]

    if max_pages is not None:
        sorted_pages = sorted_pages[:max_pages]

    total_page_count = len(page_map)
    elapsed_ms = (time.monotonic() - t_start) * 1000

    parsed_doc = ParsedDocument(
        document_id=document_id,
        filename=file_path.name,
        file_path=file_path,
        page_count=total_page_count,
        pages=sorted_pages,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
        parse_time_ms=elapsed_ms,
    )

    logger.info(
        "Docling parsed %s: %d pages, %d tables, %.0f ms",
        file_path.name,
        total_page_count,
        sum(len(p.tables) for p in sorted_pages),
        elapsed_ms,
    )
    return parsed_doc


def _convert_bbox(bbox, page_height: float | None) -> list[float] | None:
    """
    Convert a Docling BoundingBox to [x0, y0, x1, y1] top-left origin.

    Docling PDFs use CoordOrigin.BOTTOMLEFT. We convert to top-left so all
    parsers produce consistent coordinates.

    Args:
        bbox: Docling BoundingBox object (has .l, .t, .r, .b attributes).
        page_height: Page height in points. If None, we cannot convert and
                     return the raw values instead (acceptably imprecise for
                     non-visual use cases like LLM context).

    Returns:
        [x0, y0, x1, y1] in top-left origin, or None if bbox is None.
    """
    if bbox is None:
        return None

    try:

        if page_height is not None:
            # Convert bottom-left → top-left
            tl_bbox = bbox.to_top_left_origin(page_height)
            return [float(tl_bbox.l), float(tl_bbox.t), float(tl_bbox.r), float(tl_bbox.b)]
        else:
            # Return raw bbox (origin unknown but we note it in provenance)
            return [float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b)]
    except Exception as exc:
        logger.debug("Could not convert Docling bbox: %s", exc)
        return None


def _convert_table_item(
    item: "TableItem",
    page_no: int,
    bbox: list[float] | None,
    table_index: int,
) -> ParsedTable:
    """
    Convert a Docling TableItem to a ParsedTable.

    Extracts the grid (rows × cols) and optionally a pandas DataFrame.
    """
    rows: list[list[str]] = []
    cells: list[ParsedCell] = []
    dataframe = None

    # Export to DataFrame via Docling's native method
    if _PANDAS_AVAILABLE:
        try:
            dataframe = item.export_to_dataframe()
            if dataframe is not None:
                rows = [list(dataframe.columns)] + dataframe.values.tolist()
                rows = [[str(cell) for cell in row] for row in rows]
        except Exception as exc:
            logger.debug("Docling export_to_dataframe failed: %s", exc)

    # Fallback: read from item.data.table_cells
    if not rows and hasattr(item, "data") and hasattr(item.data, "table_cells"):
        try:
            max_row = max(
                (c.start_row_offset_idx for c in item.data.table_cells), default=0
            )
            max_col = max(
                (c.start_col_offset_idx for c in item.data.table_cells), default=0
            )
            grid: list[list[str]] = [
                [""] * (max_col + 1) for _ in range(max_row + 1)
            ]
            for cell in item.data.table_cells:
                r = cell.start_row_offset_idx
                c = cell.start_col_offset_idx
                grid[r][c] = cell.text or ""
                cells.append(ParsedCell(text=cell.text or "", row=r, col=c))
            rows = grid
        except Exception as exc:
            logger.debug("Docling table cell extraction failed: %s", exc)

    # Identify header row
    headers: list[str] = []
    if rows:
        from finextract.parsing.text_parser import _looks_like_header
        if _looks_like_header(rows[0]):
            headers = rows[0]

    return ParsedTable(
        page=page_no,
        bbox=bbox,
        rows=rows,
        headers=headers,
        dataframe=dataframe,
        cells=cells,
        table_index=table_index,
    )


def is_available() -> bool:
    """Return True if Docling is installed and importable."""
    return _DOCLING_AVAILABLE

"""
FinExtract-Bench: Document ingestion service.

Handles the first stage of the pipeline:
  1. Validate the input PDF (exists, readable, non-empty)
  2. Compute a deterministic document_id from the file content
  3. Detect company + fiscal year from filename / metadata (best-effort)
  4. Create a DocumentRecord in the database
  5. Optionally parse the document and return the ParsedDocument

The ingestion step is idempotent: if the same PDF is ingested twice
(same content hash), the existing DocumentRecord is returned.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from sqlalchemy.orm import Session

from finextract.parsing.base import ParsedDocument
from finextract.storage.models import DocumentRecord
from finextract.storage.repository import create_document, get_document

logger = logging.getLogger(__name__)


# ============================================================
# Document ID generation
# ============================================================


def compute_document_id(file_path: Path) -> str:
    """
    Compute a deterministic document ID from the SHA-256 of the file content.

    Using content-based IDs means the same PDF always gets the same ID,
    enabling idempotent ingestion and reproducible experiments.

    Returns:
        First 32 hex characters of the SHA-256 digest (128-bit prefix).
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in 64 KB chunks to handle large PDFs
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:32]


# ============================================================
# Metadata extraction from filename
# ============================================================


def _infer_metadata_from_filename(filename: str) -> dict[str, str | int | None]:
    """
    Best-effort extraction of company name and fiscal year from a filename.

    Examples:
        "apple_2023_10k.pdf"    → company="apple", fiscal_year=2023
        "MSFT_FY2022_AR.pdf"   → company="MSFT", fiscal_year=2022
        "report.pdf"            → company=None, fiscal_year=None

    This is a convenience fallback. Ground truth always takes precedence.
    """
    import re

    stem = Path(filename).stem

    # Look for a 4-digit year pattern
    year_match = re.search(r"(?:FY|fy)?(\b20\d{2}\b)", stem)
    fiscal_year = int(year_match.group(1)) if year_match else None

    # Strip the year and common suffixes to get company name
    company = stem
    if year_match:
        company = stem[: year_match.start()].strip("_- ")
    for suffix in ["10k", "10K", "AR", "annual", "report", "filing"]:
        company = company.replace(suffix, "").strip("_- ")

    return {
        "company": company or None,
        "fiscal_year": fiscal_year,
    }


# ============================================================
# Ingestion
# ============================================================


def ingest_document(
    session: Session,
    file_path: Path,
    *,
    company: str | None = None,
    fiscal_year: int | None = None,
    source_url: str | None = None,
    doc_type: str = "annual_report",
    parse: bool = False,
    parser: str = "text",
) -> tuple[DocumentRecord, ParsedDocument | None]:
    """
    Ingest a PDF document into the database.

    Args:
        session: Active SQLAlchemy session.
        file_path: Path to the PDF file.
        company: Company name override. If None, inferred from filename.
        fiscal_year: Fiscal year override. If None, inferred from filename.
        source_url: URL where the document was downloaded from.
        doc_type: Document type label ('annual_report', '10-K', etc.).
        parse: Whether to parse the PDF immediately after ingestion.
        parser: Parser to use if parse=True ('text' or 'docling').

    Returns:
        (DocumentRecord, ParsedDocument | None)
        ParsedDocument is None if parse=False.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If file is not a PDF.
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {file_path.suffix}")

    # ── Compute document ID ──
    t0 = time.monotonic()
    document_id = compute_document_id(file_path)
    load_ms = (time.monotonic() - t0) * 1000

    # ── Check for existing record (idempotent) ──
    existing = get_document(session, document_id)
    if existing is not None:
        logger.info("Document already ingested: %s (%s)", file_path.name, document_id)
        parsed_doc = None
        if parse:
            parsed_doc = _parse_document(file_path, document_id, parser=parser)
        return existing, parsed_doc

    # ── Infer metadata ──
    inferred = _infer_metadata_from_filename(file_path.name)
    resolved_company = company or inferred["company"] or "Unknown"
    resolved_year = fiscal_year or inferred["fiscal_year"] or 0

    # ── File metadata ──
    file_size = file_path.stat().st_size

    # ── Parse to get page count (requires opening the file) ──
    page_count: int | None = None
    parsed_doc: ParsedDocument | None = None

    if parse:
        parsed_doc = _parse_document(file_path, document_id, parser=parser)
        page_count = parsed_doc.page_count
    else:
        # Quick page count without full parse
        try:
            page_count = _get_page_count(file_path)
        except Exception as exc:
            logger.warning("Could not read page count for %s: %s", file_path.name, exc)

    # ── Create database record ──
    doc_record = DocumentRecord(
        document_id=document_id,
        company=resolved_company,
        fiscal_year=resolved_year,
        filename=file_path.name,
        file_path=str(file_path),
        file_size_bytes=file_size,
        page_count=page_count,
        doc_type=doc_type,
        source_url=source_url,
        status="ingested",
    )
    create_document(session, doc_record)

    logger.info(
        "Ingested document: %s → %s (%s, FY%s, %d pages)",
        file_path.name,
        document_id,
        resolved_company,
        resolved_year,
        page_count or 0,
    )

    return doc_record, parsed_doc


def _get_page_count(file_path: Path) -> int:
    """Open a PDF with PyMuPDF just to read the page count."""
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            return 0

    with pymupdf.open(str(file_path)) as doc:
        return len(doc)


def _parse_document(
    file_path: Path,
    document_id: str,
    parser: str,
) -> ParsedDocument:
    """Dispatch to the appropriate parser."""
    if parser == "docling":
        from finextract.parsing import docling_parser
        return docling_parser.parse_pdf(file_path, document_id)
    else:
        from finextract.parsing import text_parser
        return text_parser.parse_pdf(file_path, document_id)

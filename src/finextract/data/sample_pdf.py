"""
FinExtract-Bench: Synthetic sample PDF generator for tests.

Creates a minimal PDF that resembles a financial report section,
containing:
  - A cover page with company name and fiscal year
  - A page with an income statement table
  - A page with a balance sheet table

This PDF uses real-looking but clearly fictional data (TechCorp Inc.)
so it can never be confused with actual financial data.

The values are deterministic and match the sample ground truth used in
integration tests.

Usage::
    python scripts/create_sample_pdf.py
    # or
    from finextract.data.sample_pdf import create_sample_pdf
    path = create_sample_pdf(output_dir=Path("data/sample"))
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Sample data (fictional company, deterministic values) ────────────
SAMPLE_COMPANY = "TechCorp Inc."
SAMPLE_FISCAL_YEAR = 2023

# All values in millions USD — used in assertions in integration tests
SAMPLE_DATA = {
    "revenue": 50_000.0,
    "net_income": 10_000.0,
    "operating_income": 12_000.0,
    "total_assets": 80_000.0,
    "total_liabilities": 30_000.0,
    "cash_and_equivalents": 15_000.0,
    "eps": 5.00,
}


def create_sample_pdf(output_dir: Path | None = None) -> Path:
    """
    Create a synthetic financial report PDF for testing.

    Args:
        output_dir: Directory to write the PDF. Defaults to data/sample/.

    Returns:
        Path to the created PDF file.

    Raises:
        ImportError: If PyMuPDF is not installed.
    """
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required to create the sample PDF. "
                "Install with: pip install pymupdf"
            ) from exc

    if output_dir is None:
        # Default: data/sample relative to project root
        project_root = Path(__file__).resolve().parents[4]
        output_dir = project_root / "data" / "sample"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "techcorp_2023_annual_report.pdf"

    doc = pymupdf.open()

    _add_cover_page(doc)
    _add_income_statement_page(doc)
    _add_balance_sheet_page(doc)

    doc.save(str(out_path))
    doc.close()

    logger.info("Sample PDF created: %s", out_path)
    return out_path


def _add_cover_page(doc) -> None:
    """Add a simple cover page."""
    page = doc.new_page(width=612, height=792)  # US Letter

    page.insert_text(
        (72, 200),
        "TechCorp Inc.",
        fontsize=28,
        color=(0, 0, 0),
    )
    page.insert_text(
        (72, 250),
        "Annual Report - Fiscal Year 2023",
        fontsize=18,
        color=(0.3, 0.3, 0.3),
    )
    page.insert_text(
        (72, 300),
        "This is a synthetic document created for testing purposes only.",
        fontsize=10,
        color=(0.5, 0.5, 0.5),
    )
    page.insert_text(
        (72, 320),
        "All financial figures are fictional.",
        fontsize=10,
        color=(0.5, 0.5, 0.5),
    )


def _add_income_statement_page(doc) -> None:
    """
    Add a page with an income statement.

    The values must match SAMPLE_DATA for integration tests.
    """
    page = doc.new_page(width=612, height=792)

    page.insert_text(
        (72, 60),
        "Consolidated Statements of Operations",
        fontsize=16,
        color=(0, 0, 0),
    )
    page.insert_text(
        (72, 80),
        "For the Fiscal Year Ended December 31, 2023",
        fontsize=11,
        color=(0.3, 0.3, 0.3),
    )
    page.insert_text(
        (72, 100),
        "(in millions, except per share amounts)",
        fontsize=9,
        color=(0.5, 0.5, 0.5),
    )

    rows = [
        ("", "FY2023", "FY2022"),
        ("Revenue", "50,000", "45,000"),
        ("Cost of Revenue", "28,000", "26,000"),
        ("Gross Profit", "22,000", "19,000"),
        ("Operating Expenses", "10,000", "9,500"),
        ("Operating Income", "12,000", "9,500"),
        ("Interest and Other Income", "500", "400"),
        ("Income Before Taxes", "12,500", "9,900"),
        ("Provision for Income Taxes", "2,500", "1,900"),
        ("Net Income", "10,000", "8,000"),
        ("Earnings Per Share (Diluted)", "5.00", "4.20"),
    ]

    col_x = [72, 350, 480]
    y_start = 130
    row_height = 22

    for i, (label, fy23, fy22) in enumerate(rows):
        y = y_start + i * row_height
        page.insert_text((col_x[0], y), label, fontsize=10, color=(0, 0, 0))
        page.insert_text((col_x[1], y), fy23, fontsize=10, color=(0, 0, 0))
        page.insert_text((col_x[2], y), fy22, fontsize=10, color=(0, 0, 0))

    page.draw_line(
        (72, y_start + row_height - 5),
        (540, y_start + row_height - 5),
        color=(0, 0, 0),
        width=0.5,
    )


def _add_balance_sheet_page(doc) -> None:
    """
    Add a balance sheet page.

    Values must match SAMPLE_DATA for integration tests.
    """
    page = doc.new_page(width=612, height=792)

    page.insert_text(
        (72, 60),
        "Consolidated Balance Sheets",
        fontsize=16,
        color=(0, 0, 0),
    )
    page.insert_text(
        (72, 80),
        "As of December 31, 2023",
        fontsize=11,
        color=(0.3, 0.3, 0.3),
    )
    page.insert_text(
        (72, 100),
        "(in millions)",
        fontsize=9,
        color=(0.5, 0.5, 0.5),
    )

    rows = [
        ("", "FY2023", "FY2022"),
        ("ASSETS", "", ""),
        ("Cash and Cash Equivalents", "15,000", "12,000"),
        ("Accounts Receivable", "8,000", "7,500"),
        ("Other Current Assets", "5,000", "4,500"),
        ("Total Current Assets", "28,000", "24,000"),
        ("Property and Equipment, net", "32,000", "30,000"),
        ("Other Non-Current Assets", "20,000", "18,000"),
        ("Total Assets", "80,000", "72,000"),
        ("LIABILITIES", "", ""),
        ("Accounts Payable", "5,000", "4,800"),
        ("Other Current Liabilities", "8,000", "7,500"),
        ("Total Current Liabilities", "13,000", "12,300"),
        ("Long-Term Debt", "12,000", "11,000"),
        ("Other Long-Term Liabilities", "5,000", "4,700"),
        ("Total Liabilities", "30,000", "28,000"),
        ("Total Stockholders Equity", "50,000", "44,000"),
    ]

    col_x = [72, 350, 480]
    y_start = 130
    row_height = 22

    for i, (label, fy23, fy22) in enumerate(rows):
        y = y_start + i * row_height
        page.insert_text((col_x[0], y), label, fontsize=10, color=(0, 0, 0))
        if fy23:
            page.insert_text((col_x[1], y), fy23, fontsize=10, color=(0, 0, 0))
        if fy22:
            page.insert_text((col_x[2], y), fy22, fontsize=10, color=(0, 0, 0))

    page.draw_line(
        (72, y_start + row_height - 5),
        (540, y_start + row_height - 5),
        color=(0, 0, 0),
        width=0.5,
    )


if __name__ == "__main__":
    out = create_sample_pdf()
    print(f"Sample PDF created: {out}")

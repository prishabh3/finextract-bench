"""
FinExtract-Bench: Extraction pipelines A, B, and C.

Each pipeline orchestrates the full extraction flow for one document:
  context selection → LLM extraction → validation → provenance recording

Pipeline A — Text Only:
  Plain text from PyMuPDF → LLM → Pydantic

Pipeline B — Layout Aware:
  Docling structured doc (or PyMuPDF tables) → LLM → Pydantic

Pipeline C — Hybrid:
  Docling tables + surrounding text → LLM → Pydantic + rule-based checks

Design:
- All three pipelines return the same type: (FinancialReport, list[ProvenanceRecord]).
- Pipelines do NOT store to the database — callers handle persistence.
- Context selection is the key differentiator between pipelines.
- Max context length is capped to avoid token limits.
"""

from __future__ import annotations

import logging

from finextract.extraction.extractor import ExtractionResult, extract_from_context
from finextract.extraction.providers.base import LLMProvider
from finextract.parsing.base import ParsedDocument, ParsedTable
from finextract.provenance.tracker import ProvenanceTracker, find_source_for_value
from finextract.validation.schemas import ExtractionMethod, FinancialReport, ProvenanceRecord

logger = logging.getLogger(__name__)

# Maximum characters sent to the LLM (avoid context window overruns)
MAX_CONTEXT_CHARS = 12_000

# Financial section keywords used to identify relevant pages/tables
_INCOME_KEYWORDS = [
    "revenue", "net income", "operating income", "net sales",
    "consolidated statements of operations", "statements of earnings",
]
_BALANCE_KEYWORDS = [
    "total assets", "total liabilities", "cash and cash equivalents",
    "consolidated balance sheets",
]


# ============================================================
# Pipeline A — Text Only
# ============================================================


def run_text_only(
    parsed_doc: ParsedDocument,
    *,
    company: str,
    fiscal_year: int,
    provider: LLMProvider,
    tracker: ProvenanceTracker,
) -> tuple[FinancialReport, list[ProvenanceRecord]]:
    """
    Pipeline A: extract from plain text only.

    Context selection:
      1. Find pages containing income statement and balance sheet keywords.
      2. Concatenate their full_text up to MAX_CONTEXT_CHARS.
      3. Fallback: use all pages text if no keyword pages found.

    Args:
        parsed_doc: Parsed document from the PyMuPDF text parser.
        company: Company name for the extraction.
        fiscal_year: Target fiscal year.
        provider: LLM provider to use.
        tracker: Provenance tracker (records stage timings).

    Returns:
        (FinancialReport, list of ProvenanceRecords for each field).
    """
    with tracker.time_stage("candidate_retrieval"):
        context = _select_text_context(parsed_doc)

    logger.info(
        "Pipeline A [text_only]: %s FY%s — context %d chars",
        company, fiscal_year, len(context),
    )

    with tracker.time_stage("llm_inference"):
        result = extract_from_context(
            context,
            company=company,
            fiscal_year=fiscal_year,
            pipeline="text_only",
            provider=provider,
            document_id=parsed_doc.document_id,
        )

    provenance_records = _build_provenance_records(
        result, tracker, parsed_doc, ExtractionMethod.TEXT_ONLY
    )

    return result.report or _empty_report(company, fiscal_year, parsed_doc.document_id, ExtractionMethod.TEXT_ONLY), provenance_records


# ============================================================
# Pipeline B — Layout Aware
# ============================================================


def run_layout_aware(
    parsed_doc: ParsedDocument,
    *,
    company: str,
    fiscal_year: int,
    provider: LLMProvider,
    tracker: ProvenanceTracker,
) -> tuple[FinancialReport, list[ProvenanceRecord]]:
    """
    Pipeline B: extract using table layout information.

    Context selection:
      1. Find tables containing income statement / balance sheet keywords.
      2. Format them as structured text (header row + data rows).
      3. Include surrounding text blocks for unit context.
      4. Fallback to full text if no tables found.

    Args:
        parsed_doc: Parsed document (ideally from Docling for best table detection).
        company: Company name.
        fiscal_year: Target fiscal year.
        provider: LLM provider.
        tracker: Provenance tracker.

    Returns:
        (FinancialReport, list of ProvenanceRecords).
    """
    with tracker.time_stage("candidate_retrieval"):
        context = _select_layout_context(parsed_doc)

    logger.info(
        "Pipeline B [layout_aware]: %s FY%s — context %d chars",
        company, fiscal_year, len(context),
    )

    with tracker.time_stage("llm_inference"):
        result = extract_from_context(
            context,
            company=company,
            fiscal_year=fiscal_year,
            pipeline="layout_aware",
            provider=provider,
            document_id=parsed_doc.document_id,
        )

    provenance_records = _build_provenance_records(
        result, tracker, parsed_doc, ExtractionMethod.LAYOUT_AWARE
    )

    return result.report or _empty_report(company, fiscal_year, parsed_doc.document_id, ExtractionMethod.LAYOUT_AWARE), provenance_records


# ============================================================
# Pipeline C — Hybrid
# ============================================================


def run_hybrid(
    parsed_doc: ParsedDocument,
    *,
    company: str,
    fiscal_year: int,
    provider: LLMProvider,
    tracker: ProvenanceTracker,
) -> tuple[FinancialReport, list[ProvenanceRecord]]:
    """
    Pipeline C: richest context — tables + surrounding text + unit context.

    Context selection:
      1. Find financial tables with keyword matching.
      2. Include table text + preceding text blocks for unit/header context.
      3. Apply rule-based unit consistency checks post-extraction.

    Args:
        parsed_doc: Parsed document (ideally Docling for table structure).
        company: Company name.
        fiscal_year: Target fiscal year.
        provider: LLM provider.
        tracker: Provenance tracker.

    Returns:
        (FinancialReport, list of ProvenanceRecords).
    """
    with tracker.time_stage("candidate_retrieval"):
        context = _select_hybrid_context(parsed_doc)

    logger.info(
        "Pipeline C [hybrid]: %s FY%s — context %d chars",
        company, fiscal_year, len(context),
    )

    with tracker.time_stage("llm_inference"):
        result = extract_from_context(
            context,
            company=company,
            fiscal_year=fiscal_year,
            pipeline="hybrid",
            provider=provider,
            document_id=parsed_doc.document_id,
        )

    # ── Rule-based post-processing (hybrid only) ─────────────────────
    if result.report:
        _apply_consistency_checks(result.report, context)

    provenance_records = _build_provenance_records(
        result, tracker, parsed_doc, ExtractionMethod.HYBRID
    )

    return result.report or _empty_report(company, fiscal_year, parsed_doc.document_id, ExtractionMethod.HYBRID), provenance_records


# ============================================================
# Context selection helpers
# ============================================================


def _select_text_context(doc: ParsedDocument) -> str:
    """
    Select relevant pages based on financial keywords and concatenate their text.
    """
    relevant_pages = []
    all_text_pages = []

    for page in doc.pages:
        text_lower = page.full_text.lower()
        all_text_pages.append(page.full_text)
        is_relevant = any(kw in text_lower for kw in _INCOME_KEYWORDS + _BALANCE_KEYWORDS)
        if is_relevant:
            relevant_pages.append(page.full_text)

    # Use relevant pages if found, fallback to all pages
    source_pages = relevant_pages if relevant_pages else all_text_pages
    context = "\n\n--- PAGE BREAK ---\n\n".join(source_pages)

    return context[:MAX_CONTEXT_CHARS]


def _select_layout_context(doc: ParsedDocument) -> str:
    """
    Format detected tables as structured text, focusing on financial tables.
    Include surrounding text for unit context.
    """
    parts: list[str] = []
    all_tables = doc.all_tables()

    if not all_tables:
        # No tables detected — fallback to text pipeline logic
        logger.debug("No tables detected; falling back to text context for layout_aware.")
        return _select_text_context(doc)

    for table in all_tables:
        # Filter for tables that look financial
        table_text = _table_to_text(table)
        table_lower = table_text.lower()
        if any(kw in table_lower for kw in _INCOME_KEYWORDS + _BALANCE_KEYWORDS):
            # Add surrounding page text for unit context (first 200 chars)
            page = doc.get_page(table.page)
            if page:
                page_header = page.full_text[:300]
                if page_header.strip():
                    parts.append(f"[Page {table.page} context]\n{page_header}")
            parts.append(f"[Table on page {table.page}]\n{table_text}")

    if not parts:
        # No financial tables found — fallback
        logger.debug("No financial tables matched; falling back to text context.")
        return _select_text_context(doc)

    context = "\n\n".join(parts)
    return context[:MAX_CONTEXT_CHARS]


def _select_hybrid_context(doc: ParsedDocument) -> str:
    """
    Combine table context with full page text for pages containing financial data.
    This gives the LLM both structured table content and surrounding narrative.
    """
    parts: list[str] = []
    pages_included: set[int] = set()

    # First pass: include financial table pages with full surrounding text
    for table in doc.all_tables():
        table_text = _table_to_text(table)
        table_lower = table_text.lower()
        if any(kw in table_lower for kw in _INCOME_KEYWORDS + _BALANCE_KEYWORDS):
            page_num = table.page
            if page_num not in pages_included:
                page = doc.get_page(page_num)
                if page:
                    parts.append(
                        f"[Page {page_num} — full text]\n{page.full_text[:1500]}"
                    )
                pages_included.add(page_num)
            parts.append(f"[Table on page {table.page}]\n{table_text}")

    # Second pass: add keyword-matching pages not already included
    for page in doc.pages:
        if page.page_number in pages_included:
            continue
        text_lower = page.full_text.lower()
        if any(kw in text_lower for kw in _INCOME_KEYWORDS + _BALANCE_KEYWORDS):
            parts.append(f"[Page {page.page_number}]\n{page.full_text[:2000]}")
            pages_included.add(page.page_number)

    if not parts:
        return _select_text_context(doc)

    context = "\n\n".join(parts)
    return context[:MAX_CONTEXT_CHARS]


def _table_to_text(table: ParsedTable) -> str:
    """
    Format a ParsedTable as pipe-delimited text for LLM consumption.

    Example output:
        | Revenue | 50,000 | 45,000 |
        | Net Income | 10,000 | 8,000 |
    """
    if not table.rows:
        return ""
    lines = []
    for row in table.rows:
        line = " | ".join(str(cell) for cell in row)
        lines.append(f"| {line} |")
    return "\n".join(lines)


# ============================================================
# Rule-based consistency checks (hybrid only)
# ============================================================


def _apply_consistency_checks(report: FinancialReport, context: str) -> None:
    """
    Apply deterministic sanity checks to a hybrid extraction result.

    Modifies report in-place. Adds warnings to validation_errors without
    invalidating the report (they are informational, not blocking).

    Current checks:
      1. Revenue should be >= operating_income (if both present).
      2. Total assets should be >= total_liabilities (if both present).
      3. Revenue should be > 0 for a normal for-profit company.
    """
    errors = list(report.validation_errors)

    # Check 1: revenue >= operating_income
    if report.revenue and report.operating_income:
        rev = report.revenue.value
        op = report.operating_income.value
        if rev > 0 and op > rev:
            errors.append(
                f"Consistency warning: operating_income ({op}) > revenue ({rev}). "
                "This may indicate a unit mismatch or wrong table selection."
            )

    # Check 2: total_assets >= total_liabilities
    if report.total_assets and report.total_liabilities:
        assets = report.total_assets.value
        liabs = report.total_liabilities.value
        if assets > 0 and liabs > assets:
            errors.append(
                f"Consistency warning: total_liabilities ({liabs}) > total_assets ({assets}). "
                "Verify values are from the same Balance Sheet."
            )

    # Check 3: revenue > 0
    if report.revenue and report.revenue.value <= 0:
        errors.append(
            f"Consistency warning: revenue ({report.revenue.value}) is non-positive. "
            "Verify the correct row was extracted."
        )

    if errors != report.validation_errors:
        # We're modifying a Pydantic model with validate_assignment=True
        report.validation_errors = errors


# ============================================================
# Provenance record construction
# ============================================================


def _build_provenance_records(
    result: ExtractionResult,
    tracker: ProvenanceTracker,
    parsed_doc: ParsedDocument,
    extraction_method: ExtractionMethod,
) -> list[ProvenanceRecord]:
    """
    Build one ProvenanceRecord per extracted field.

    For each field in the report, we try to locate the value in the
    parsed document to get page and source_text provenance.
    """
    records: list[ProvenanceRecord] = []
    report = result.report

    if report is None:
        return records

    llm_resp = result.llm_response

    for field_name, metric in report.metric_fields().items():
        if metric is None:
            continue

        # Try to find source location for this value in the parsed doc
        loc = find_source_for_value(
            parsed_doc,
            str(metric.original_value or metric.value),
        )

        record = tracker.build_record(
            field_name=field_name,
            extraction_method=extraction_method,
            page=metric.page or loc.get("page"),
            source_text=metric.source_text or loc.get("source_text"),
            bbox=metric.bbox or loc.get("bbox"),
            confidence=metric.confidence,
            input_tokens=llm_resp.input_tokens,
            output_tokens=llm_resp.output_tokens,
        )
        records.append(record)

    return records


# ============================================================
# Helper: empty report on total failure
# ============================================================


def _empty_report(
    company: str,
    fiscal_year: int,
    document_id: str | None,
    method: ExtractionMethod,
) -> FinancialReport:
    """Return a minimal INVALID FinancialReport when extraction totally fails."""
    from finextract.validation.schemas import ValidationStatus

    return FinancialReport(
        company=company,
        fiscal_year=fiscal_year,
        document_id=document_id,
        extraction_method=method,
        validation_status=ValidationStatus.INVALID,
        validation_errors=["Extraction failed to produce a parseable result."],
    )

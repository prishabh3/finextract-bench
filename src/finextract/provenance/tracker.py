"""
FinExtract-Bench: Provenance tracking for extracted financial metrics.

Every extracted metric should have a ProvenanceRecord that answers:
  - WHERE did this value come from? (document, page, bbox, source_text)
  - HOW was it extracted? (parser, LLM, pipeline)
  - WHO extracted it? (provider, model, version)
  - WHEN? (timestamp, processing time)
  - AT WHAT COST? (tokens, estimated USD)

This module provides a ProvenanceTracker that accumulates timing
measurements across pipeline stages and builds the final record.

Design: The tracker is a context manager that records wall-clock time
for each named stage. Timings are stored as milliseconds.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator

from finextract.parsing.base import ParsedDocument, ParsedTable
from finextract.validation.schemas import ExtractionMethod, ProvenanceRecord

logger = logging.getLogger(__name__)


# ============================================================
# Stage timing
# ============================================================


@dataclass
class StageTimer:
    """Accumulated wall-clock timings for each pipeline stage (ms)."""

    pdf_loading: float | None = None
    pdf_parsing: float | None = None
    candidate_retrieval: float | None = None
    llm_inference: float | None = None
    validation: float | None = None
    storage: float | None = None

    @property
    def total(self) -> float:
        """Sum of all non-None stage times."""
        stages = [
            self.pdf_loading,
            self.pdf_parsing,
            self.candidate_retrieval,
            self.llm_inference,
            self.validation,
            self.storage,
        ]
        return sum(s for s in stages if s is not None)


# ============================================================
# Provenance tracker
# ============================================================


class ProvenanceTracker:
    """
    Tracks timing and metadata across pipeline stages for one document.

    Usage::

        tracker = ProvenanceTracker(document_id="...", company="Apple", ...)

        with tracker.time_stage("pdf_loading"):
            doc = pymupdf.open(...)

        with tracker.time_stage("llm_inference"):
            response = llm.extract(...)

        record = tracker.build_record(
            field_name="revenue",
            extraction_method=ExtractionMethod.TEXT_ONLY,
            ...
        )
    """

    def __init__(
        self,
        document_id: str,
        company: str,
        fiscal_year: int,
        parser_name: str | None = None,
        parser_version: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        self.document_id = document_id
        self.company = company
        self.fiscal_year = fiscal_year
        self.parser_name = parser_name
        self.parser_version = parser_version
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.timer = StageTimer()
        self._started_at = datetime.utcnow()

        # Set from parsed_document if provided later
        self._parsed_doc: ParsedDocument | None = None

    def set_parsed_document(self, parsed_doc: ParsedDocument) -> None:
        """
        Attach the parsed document so the tracker can infer parser metadata.
        Also records parse time from the document's parse_time_ms field.
        """
        self._parsed_doc = parsed_doc
        self.parser_name = parsed_doc.parser_name
        self.parser_version = parsed_doc.parser_version
        if parsed_doc.parse_time_ms is not None:
            self.timer.pdf_parsing = parsed_doc.parse_time_ms

    @contextmanager
    def time_stage(self, stage: str) -> Generator[None, None, None]:
        """
        Context manager to time a named pipeline stage.

        Valid stages: 'pdf_loading', 'pdf_parsing', 'candidate_retrieval',
                      'llm_inference', 'validation', 'storage'

        Usage::

            with tracker.time_stage("llm_inference"):
                result = llm.call(...)
        """
        t_start = time.monotonic()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic() - t_start) * 1000
            self._set_stage_time(stage, elapsed_ms)

    def _set_stage_time(self, stage: str, ms: float) -> None:
        if stage == "pdf_loading":
            self.timer.pdf_loading = ms
        elif stage == "pdf_parsing":
            self.timer.pdf_parsing = ms
        elif stage == "candidate_retrieval":
            self.timer.candidate_retrieval = ms
        elif stage == "llm_inference":
            self.timer.llm_inference = ms
        elif stage == "validation":
            self.timer.validation = ms
        elif stage == "storage":
            self.timer.storage = ms
        else:
            logger.warning("Unknown stage name: %r (timing discarded)", stage)

    def set_stage_time(self, stage: str, ms: float) -> None:
        """Manually set a stage time (use when not using the context manager)."""
        self._set_stage_time(stage, ms)

    def build_record(
        self,
        *,
        field_name: str,
        extraction_method: ExtractionMethod,
        page: int | None = None,
        source_text: str | None = None,
        bbox: list[float] | None = None,
        confidence: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
    ) -> ProvenanceRecord:
        """
        Build a ProvenanceRecord from the accumulated tracking state.

        Call once per extracted field after all pipeline stages complete.

        Args:
            field_name: Name of the field being recorded (e.g. 'revenue').
            extraction_method: Which pipeline produced this extraction.
            page: 1-indexed page where the value was found (None if unknown).
            source_text: Verbatim text snippet (None if unavailable).
            bbox: Bounding box [x0, y0, x1, y1] (None if unavailable).
            confidence: Model confidence [0, 1] (None if not applicable).
            input_tokens: LLM input token count (None if mock/unavailable).
            output_tokens: LLM output token count.
            estimated_cost_usd: Estimated USD cost for this field extraction.

        Returns:
            ProvenanceRecord with all available provenance data.
        """
        return ProvenanceRecord(
            provenance_id=str(uuid.uuid4()),
            document_id=self.document_id,
            company=self.company,
            fiscal_year=self.fiscal_year,
            field_name=field_name,
            # Source location
            page=page,
            source_text=source_text,
            bbox=bbox,
            # Extraction metadata
            extraction_method=extraction_method,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            # Quality / cost
            confidence=confidence,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd,
            # Timing
            extracted_at=self._started_at,
            processing_time_ms=self.timer.total,
            time_pdf_loading_ms=self.timer.pdf_loading,
            time_pdf_parsing_ms=self.timer.pdf_parsing,
            time_candidate_retrieval_ms=self.timer.candidate_retrieval,
            time_llm_inference_ms=self.timer.llm_inference,
            time_validation_ms=self.timer.validation,
            time_storage_ms=self.timer.storage,
        )


# ============================================================
# Source location helpers
# ============================================================


def find_source_for_value(
    parsed_doc: ParsedDocument,
    value_str: str,
    *,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """
    Search a ParsedDocument for the page and source text containing value_str.

    Returns a dict with keys: page, source_text, bbox.
    All values are None if the string is not found.

    This is best-effort — if the same string appears multiple times,
    the first occurrence is returned.
    """
    search_val = value_str if case_sensitive else value_str.lower()

    for page in parsed_doc.pages:
        text = page.full_text if case_sensitive else page.full_text.lower()
        if search_val in text:
            # Find the surrounding context
            idx = text.find(search_val)
            start = max(0, idx - 100)
            end = min(len(page.full_text), idx + len(value_str) + 100)
            return {
                "page": page.page_number,
                "source_text": page.full_text[start:end].strip(),
                "bbox": None,  # Page-level scan can't give a precise bbox
            }

        # Also search within individual text blocks for a bbox
        for block in page.text_blocks:
            block_text = block.text if case_sensitive else block.text.lower()
            if search_val in block_text:
                return {
                    "page": page.page_number,
                    "source_text": block.text[:500],
                    "bbox": block.bbox,
                }

    return {"page": None, "source_text": None, "bbox": None}


def find_table_containing_keyword(
    parsed_doc: ParsedDocument,
    keyword: str,
    *,
    case_sensitive: bool = False,
) -> ParsedTable | None:
    """
    Find the first table whose text contains the given keyword.

    Used by the layout-aware and hybrid pipelines to locate the income
    statement or balance sheet table.
    """
    search_kw = keyword if case_sensitive else keyword.lower()

    for page in parsed_doc.pages:
        for table in page.tables:
            table_text = " ".join(
                " ".join(str(cell) for cell in row)
                for row in table.rows
            )
            if not case_sensitive:
                table_text = table_text.lower()
            if search_kw in table_text:
                return table

    return None

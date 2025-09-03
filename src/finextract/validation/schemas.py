"""
FinExtract-Bench: Pydantic v2 data schemas for financial document extraction.

Design principles:
- Strict validation — invalid data never enters the evaluation database silently.
- Provenance-first — every metric carries the information needed to audit it.
- Extensible — adding a new financial field requires adding one attribute; no
  pipeline logic needs changing.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

# ============================================================
# Enumerations
# ============================================================


class ExtractionMethod(str, Enum):
    """Which pipeline produced the extraction."""

    TEXT_ONLY = "text_only"
    LAYOUT_AWARE = "layout_aware"
    HYBRID = "hybrid"
    MOCK = "mock"
    UNKNOWN = "unknown"


class ValidationStatus(str, Enum):
    """Result of Pydantic + rule-based validation."""

    VALID = "valid"
    PARTIAL = "partial"  # Some fields missing but extractable fields are valid
    INVALID = "invalid"  # One or more fields failed validation


class FailureType(str, Enum):
    """Formal failure taxonomy — see evaluation/failure.py for classifier logic."""

    WRONG_TABLE = "WRONG_TABLE"
    COLUMN_SHIFT = "COLUMN_SHIFT"
    PAGE_BOUNDARY = "PAGE_BOUNDARY"
    UNIT_NORMALIZATION = "UNIT_NORMALIZATION"
    CURRENCY_NORMALIZATION = "CURRENCY_NORMALIZATION"
    SIGN_ERROR = "SIGN_ERROR"
    OCR_ERROR = "OCR_ERROR"
    MISSING_VALUE = "MISSING_VALUE"
    SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"
    DUPLICATE_VALUE = "DUPLICATE_VALUE"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    OTHER = "OTHER"


class FailureSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# Core metric schema
# ============================================================


class FinancialMetric(BaseModel):
    """
    A single extracted financial metric with full provenance.

    All provenance fields are Optional — if a parser cannot provide a value
    (e.g. text-only extraction has no bounding box), store None rather than
    fabricating data.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    # Core value
    value: float = Field(description="Normalized numeric value.")
    original_value: str | None = Field(
        default=None,
        description="Raw string as it appeared in the document before normalization.",
    )
    unit: str = Field(
        default="USD",
        description="Unit of measurement, e.g. 'USD', 'million USD', 'USD per share'.",
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 currency code, e.g. 'USD', 'EUR'.",
    )

    # Provenance — all optional; None means 'not available', never fabricated
    page: int | None = Field(default=None, ge=1, description="1-indexed page number.")
    source_text: str | None = Field(
        default=None,
        max_length=2000,
        description="Verbatim text snippet from which the value was extracted.",
    )
    bbox: list[float] | None = Field(
        default=None,
        description="Bounding box [x0, y0, x1, y1] in page coordinates.",
    )
    confidence: float | None = Field(
        default=None,
        description="Extraction confidence in [0.0, 1.0] — None if not computable.",
    )
    extraction_method: ExtractionMethod = Field(default=ExtractionMethod.UNKNOWN)

    # ---- Validators ----

    @field_validator("currency", mode="before")
    @classmethod
    def _validate_currency(cls, v: Any) -> Any:
        """Enforce ISO 4217 format (3 uppercase letters) when provided."""
        if v is None:
            return v
        if isinstance(v, str):
            v = v.strip().upper()
            if not re.match(r"^[A-Z]{3}$", v):
                raise ValueError(
                    f"Currency must be a 3-letter ISO 4217 code, got: {v!r}"
                )
        return v

    @field_validator("bbox", mode="after")
    @classmethod
    def _validate_bbox(cls, v: list[float] | None) -> list[float] | None:
        """Bounding box must have exactly 4 coords and form a valid rectangle."""
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError(f"Bounding box must have exactly 4 values, got {len(v)}.")
        x0, y0, x1, y1 = v
        if x0 >= x1:
            raise ValueError(f"Bounding box x0 ({x0}) must be < x1 ({x1}).")
        if y0 >= y1:
            raise ValueError(f"Bounding box y0 ({y0}) must be < y1 ({y1}).")
        return v

    @field_validator("confidence", mode="after")
    @classmethod
    def _validate_confidence(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {v}.")
        return v

    @field_validator("value", mode="before")
    @classmethod
    def _value_must_be_finite(cls, v: Any) -> Any:
        """Reject NaN and Inf — these indicate a normalization bug."""
        import math

        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("Metric value must be a finite number.")
        return v


# ============================================================
# Top-level report schema
# ============================================================


class FinancialReport(BaseModel):
    """
    Structured extraction result for one company/fiscal-year pair.

    All financial fields are Optional — a missing field means the pipeline
    could not extract it, which is a measurable outcome in evaluation.

    To add a new field:
      1. Add it here as `field_name: FinancialMetric | None = None`
      2. Add it to the ground-truth CSV
      3. The evaluation harness auto-discovers fields via model_fields
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    # Identity
    company: str = Field(description="Company name as it appears in the filing.")
    fiscal_year: int = Field(
        ge=1900,
        le=2100,
        description="Fiscal year of the annual report.",
    )
    document_id: str | None = Field(
        default=None,
        description="Internal document identifier from the ingestion step.",
    )

    # Financial metrics — add new fields here
    revenue: FinancialMetric | None = None
    net_income: FinancialMetric | None = None
    operating_income: FinancialMetric | None = None
    total_assets: FinancialMetric | None = None
    total_liabilities: FinancialMetric | None = None
    cash_and_equivalents: FinancialMetric | None = None
    eps: FinancialMetric | None = None

    # Extraction metadata
    extraction_method: ExtractionMethod = Field(default=ExtractionMethod.UNKNOWN)
    validation_status: ValidationStatus = Field(default=ValidationStatus.VALID)
    validation_errors: list[str] = Field(default_factory=list)
    extracted_at: datetime | None = Field(default=None)

    @field_validator("fiscal_year", mode="before")
    @classmethod
    def _coerce_fiscal_year(cls, v: Any) -> Any:
        """Accept string years like '2023' as well as integers."""
        if isinstance(v, str):
            v = v.strip()
            if re.match(r"^\d{4}$", v):
                return int(v)
            raise ValueError(f"fiscal_year string must be a 4-digit year, got: {v!r}")
        return v

    def metric_fields(self) -> dict[str, FinancialMetric | None]:
        """
        Return a dict of {field_name: metric} for all financial metric fields.

        Used by the evaluation harness to iterate over extractable fields
        without hard-coding field names.
        """
        metric_field_names = {
            "revenue",
            "net_income",
            "operating_income",
            "total_assets",
            "total_liabilities",
            "cash_and_equivalents",
            "eps",
        }
        return {k: getattr(self, k) for k in metric_field_names}

    def extraction_coverage(self) -> float:
        """Fraction of metric fields that were successfully extracted (not None)."""
        metrics = self.metric_fields()
        if not metrics:
            return 0.0
        extracted = sum(1 for v in metrics.values() if v is not None)
        return extracted / len(metrics)


# ============================================================
# Provenance record
# ============================================================


class ProvenanceRecord(BaseModel):
    """
    Detailed audit trail for a single extraction event.

    Stored in the database alongside every extracted metric so experiments
    can be reproduced and failures can be traced to their root cause.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    # Identity
    provenance_id: str | None = Field(default=None, description="UUID assigned at storage time.")
    document_id: str = Field(description="ID of the source document.")
    company: str
    fiscal_year: int = Field(ge=1900, le=2100)
    field_name: str = Field(description="Name of the extracted field, e.g. 'revenue'.")

    # Source location
    page: int | None = Field(default=None, ge=1)
    source_text: str | None = Field(default=None, max_length=2000)
    bbox: list[float] | None = None

    # Extraction metadata
    extraction_method: ExtractionMethod = Field(default=ExtractionMethod.UNKNOWN)
    parser_name: str | None = Field(default=None, description="e.g. 'docling', 'pymupdf'.")
    parser_version: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None

    # Quality / cost
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)

    # Timing
    extracted_at: datetime | None = None
    processing_time_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="End-to-end processing time in milliseconds.",
    )

    # Stage-level timings (all in milliseconds)
    time_pdf_loading_ms: float | None = None
    time_pdf_parsing_ms: float | None = None
    time_candidate_retrieval_ms: float | None = None
    time_llm_inference_ms: float | None = None
    time_validation_ms: float | None = None
    time_storage_ms: float | None = None


# ============================================================
# Evaluation schemas
# ============================================================


class FieldComparison(BaseModel):
    """Comparison of a single extracted field against ground truth."""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    predicted_value: float | None
    ground_truth_value: float | None
    absolute_error: float | None = None
    relative_error: float | None = None

    # Correctness at various tolerances
    exact_match: bool | None = None
    within_05pct: bool | None = None
    within_1pct: bool | None = None
    within_5pct: bool | None = None

    predicted_unit: str | None = None
    ground_truth_unit: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _compute_errors(self) -> Self:
        """Auto-compute error metrics when both values are present."""
        p, g = self.predicted_value, self.ground_truth_value
        if p is None or g is None:
            return self
        self.absolute_error = abs(p - g)
        self.relative_error = self.absolute_error / abs(g) if g != 0 else None
        self.exact_match = p == g
        if self.relative_error is not None:
            self.within_05pct = self.relative_error <= 0.005
            self.within_1pct = self.relative_error <= 0.01
            self.within_5pct = self.relative_error <= 0.05
        return self


class DocumentEvaluation(BaseModel):
    """Evaluation summary for one document/pipeline combination."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    company: str
    fiscal_year: int
    pipeline: ExtractionMethod
    field_comparisons: list[FieldComparison] = Field(default_factory=list)

    # Aggregate metrics (populated by the evaluation harness)
    extraction_coverage: float | None = None
    exact_accuracy: float | None = None
    accuracy_05pct: float | None = None
    accuracy_1pct: float | None = None
    accuracy_5pct: float | None = None
    validation_failure_rate: float | None = None
    evaluated_at: datetime | None = None


class FailureRecord(BaseModel):
    """A single classified extraction failure for failure analysis."""

    model_config = ConfigDict(extra="forbid")

    failure_id: str | None = None
    document_id: str
    field: str
    expected_value: float | None
    predicted_value: float | None
    failure_type: FailureType = FailureType.OTHER
    root_cause: str | None = None
    pipeline: ExtractionMethod
    page: int | None = None
    source_text: str | None = None
    severity: FailureSeverity = FailureSeverity.MEDIUM
    notes: str | None = None
    auto_classified: bool = Field(
        default=True,
        description="False if a human reviewed/corrected the classification.",
    )

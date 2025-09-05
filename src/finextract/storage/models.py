"""
FinExtract-Bench: SQLAlchemy 2.x ORM models.

Design notes:
- Uses DeclarativeBase and Mapped[T] = mapped_column(...) (SQLAlchemy 2.x style).
- All foreign keys have ON DELETE CASCADE.
- Schema is designed so that SQLite can be swapped for PostgreSQL with minimal
  changes (only the database URL needs updating).
- JSON fields use sqlalchemy.JSON which maps to TEXT in SQLite and JSONB in Postgres.
- All datetime fields are stored as UTC.
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ============================================================
# Base class
# ============================================================


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


# ============================================================
# Helper: UTC now
# ============================================================


def _utcnow() -> datetime:
    """Return the current UTC time (timezone-naive for SQLite compatibility)."""
    return datetime.utcnow()


# ============================================================
# Table: documents
# ============================================================


class DocumentRecord(Base):
    """
    One ingested PDF document.

    A document belongs to one company/fiscal-year pair.  Multiple pipelines
    can produce extraction records for the same document.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False,
        comment="UUID assigned at ingestion.",
    )
    company: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doc_type: Mapped[str] = mapped_column(
        String(50), default="annual_report",
        comment="Document type: annual_report, 10-K, etc.",
    )
    source_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), default="ingested",
        comment="Lifecycle status: ingested | parsed | extracted | evaluated.",
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )

    # Relationships
    extractions: Mapped[list[ExtractionRecord]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    evaluations: Mapped[list[EvaluationRecord]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_documents_company_year", "company", "fiscal_year"),
    )

    def __repr__(self) -> str:
        return f"<Document {self.document_id} — {self.company} {self.fiscal_year}>"


# ============================================================
# Table: extractions
# ============================================================


class ExtractionRecord(Base):
    """
    A complete structured extraction result for one document/pipeline run.

    The `result_json` column stores the full FinancialReport as JSON so
    that raw results are always preserved for debugging.
    """

    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    extraction_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pipeline: Mapped[str] = mapped_column(
        String(50), nullable=False,
        comment="text_only | layout_aware | hybrid | mock",
    )
    validation_status: Mapped[str] = mapped_column(
        String(20), default="valid",
        comment="valid | partial | invalid",
    )
    extraction_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    result_json: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Full FinancialReport serialized as JSON.",
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    # Relationships
    document: Mapped[DocumentRecord] = relationship(back_populates="extractions")
    metrics: Mapped[list[MetricRecord]] = relationship(
        back_populates="extraction",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    provenance_records: Mapped[list[ProvenanceRecord]] = relationship(
        back_populates="extraction",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    failure_records: Mapped[list[FailureRecord]] = relationship(
        back_populates="extraction",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_extractions_doc_pipeline", "document_id", "pipeline"),
    )

    def __repr__(self) -> str:
        return f"<Extraction {self.extraction_id} [{self.pipeline}]>"


# ============================================================
# Table: metrics
# ============================================================


class MetricRecord(Base):
    """
    One extracted financial metric with its normalized value and provenance.

    A row in this table corresponds to one FinancialMetric in a FinancialReport.
    """

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    extraction_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("extractions.extraction_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    field_name: Mapped[str] = mapped_column(
        String(100), nullable=False,
        comment="e.g. 'revenue', 'net_income'",
    )
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    original_value: Mapped[str | None] = mapped_column(String(500), nullable=True)
    unit: Mapped[str | None] = mapped_column(String(100), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(10), nullable=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    bbox_json: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
        comment="[x0, y0, x1, y1] as JSON string.",
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extraction_method: Mapped[str | None] = mapped_column(String(50), nullable=True)

    extraction: Mapped[ExtractionRecord] = relationship(back_populates="metrics")

    __table_args__ = (
        Index("ix_metrics_extraction_field", "extraction_id", "field_name"),
    )

    def bbox(self) -> list[float] | None:
        """Deserialize bbox from JSON string."""
        if self.bbox_json is None:
            return None
        return json.loads(self.bbox_json)


# ============================================================
# Table: provenance
# ============================================================


class ProvenanceRecord(Base):
    """Detailed audit trail for each extraction event."""

    __tablename__ = "provenance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provenance_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    extraction_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("extractions.extraction_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    company: Mapped[str] = mapped_column(String(255), nullable=False)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    field_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Source location
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    bbox_json: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Extraction metadata
    extraction_method: Mapped[str | None] = mapped_column(String(50), nullable=True)
    parser_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    llm_provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # Quality / cost
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Timing (all in milliseconds)
    extracted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    processing_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_pdf_loading_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_pdf_parsing_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_candidate_retrieval_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_llm_inference_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_validation_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_storage_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    extraction: Mapped[ExtractionRecord] = relationship(back_populates="provenance_records")


# ============================================================
# Table: evaluations
# ============================================================


class EvaluationRecord(Base):
    """Per-document evaluation result for one pipeline."""

    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    experiment_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    pipeline: Mapped[str] = mapped_column(String(50), nullable=False)

    # Aggregate accuracy metrics
    extraction_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    exact_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy_05pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy_1pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy_5pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_failure_rate: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Field-level comparisons stored as JSON
    field_comparisons_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    document: Mapped[DocumentRecord] = relationship(back_populates="evaluations")

    __table_args__ = (
        Index("ix_evaluations_doc_pipeline", "document_id", "pipeline"),
    )


# ============================================================
# Table: failures
# ============================================================


class FailureRecord(Base):
    """A classified extraction failure."""

    __tablename__ = "failures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    failure_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    extraction_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("extractions.extraction_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    field: Mapped[str] = mapped_column(String(100), nullable=False)
    expected_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    failure_type: Mapped[str] = mapped_column(String(50), nullable=False)
    root_cause: Mapped[str | None] = mapped_column(Text, nullable=True)
    pipeline: Mapped[str] = mapped_column(String(50), nullable=False)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    severity: Mapped[str] = mapped_column(String(20), default="medium")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    auto_classified: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    extraction: Mapped[ExtractionRecord] = relationship(back_populates="failure_records")

    __table_args__ = (
        Index("ix_failures_doc_field", "document_id", "field"),
    )


# ============================================================
# Table: experiments
# ============================================================


class ExperimentRecord(Base):
    """
    Metadata for a complete experiment run.

    One experiment = one (pipeline, dataset) combination run to completion.
    """

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    pipeline: Mapped[str] = mapped_column(String(50), nullable=False)
    dataset: Mapped[str] = mapped_column(String(100), nullable=False)
    llm_provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # Aggregate experiment metrics
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    total_fields: Mapped[int] = mapped_column(Integer, default=0)
    exact_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy_1pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    median_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    p95_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_per_document_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    config_json: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Full experiment config as JSON for reproducibility.",
    )
    result_file: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default="running",
        comment="running | completed | failed",
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Experiment {self.experiment_id} [{self.pipeline}/{self.dataset}]>"

"""
FinExtract-Bench: Database access layer (Repository pattern).

All database I/O goes through this module. Route handlers and service
modules should call these functions rather than touching SQLAlchemy directly.

Uses SQLAlchemy 2.x synchronous API (suitable for SQLite + FastAPI without
requiring async drivers). Can be upgraded to async with create_async_engine
and AsyncSession when switching to PostgreSQL.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from finextract.config.settings import settings
from finextract.storage.models import (
    Base,
    DocumentRecord,
    EvaluationRecord,
    ExperimentRecord,
    ExtractionRecord,
    FailureRecord,
    MetricRecord,
    ProvenanceRecord,
)

logger = logging.getLogger(__name__)


# ============================================================
# Engine + session factory
# ============================================================


def _create_engine_from_settings():
    """Create a SQLAlchemy engine from application settings."""
    url = settings.database_url
    connect_args: dict[str, Any] = {}

    if url.startswith("sqlite"):
        # Required for SQLite to enforce foreign-key constraints
        connect_args["check_same_thread"] = False

    return create_engine(
        url,
        connect_args=connect_args,
        echo=False,  # Set to True during debugging to see SQL
    )


_engine = _create_engine_from_settings()
_SessionFactory = sessionmaker(bind=_engine, autoflush=True, expire_on_commit=False)


def get_engine():
    """Return the application-wide SQLAlchemy engine."""
    return _engine


def init_db() -> None:
    """
    Create all tables if they do not exist.

    Safe to call multiple times (idempotent). Call once at application startup.
    """
    Base.metadata.create_all(bind=_engine)
    logger.info("Database tables initialized.")


def drop_all_tables() -> None:
    """
    Drop all tables. USE ONLY IN TESTS.
    """
    Base.metadata.drop_all(bind=_engine)
    logger.warning("All database tables dropped.")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that yields a transactional Session.

    Commits on clean exit; rolls back on exception.

    Usage::

        with get_session() as session:
            session.add(record)
    """
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ============================================================
# Document CRUD
# ============================================================


def create_document(session: Session, record: DocumentRecord) -> DocumentRecord:
    """Persist a new DocumentRecord and return it with its assigned id."""
    session.add(record)
    session.flush()
    logger.debug("Created document %s", record.document_id)
    return record


def get_document(session: Session, document_id: str) -> DocumentRecord | None:
    """Fetch a document by its UUID."""
    stmt = select(DocumentRecord).where(DocumentRecord.document_id == document_id)
    return session.scalars(stmt).first()


def list_documents(session: Session, *, limit: int = 100) -> list[DocumentRecord]:
    """Return the most recently ingested documents."""
    stmt = select(DocumentRecord).order_by(DocumentRecord.ingested_at.desc()).limit(limit)
    return list(session.scalars(stmt).all())


def update_document_status(session: Session, document_id: str, status: str) -> None:
    """Update the lifecycle status of a document."""
    doc = get_document(session, document_id)
    if doc is None:
        raise ValueError(f"Document not found: {document_id}")
    doc.status = status
    session.flush()


# ============================================================
# Extraction CRUD
# ============================================================


def create_extraction(session: Session, record: ExtractionRecord) -> ExtractionRecord:
    """Persist a new ExtractionRecord."""
    session.add(record)
    session.flush()
    logger.debug("Created extraction %s", record.extraction_id)
    return record


def get_extraction(session: Session, extraction_id: str) -> ExtractionRecord | None:
    """Fetch an extraction by its UUID."""
    stmt = select(ExtractionRecord).where(
        ExtractionRecord.extraction_id == extraction_id
    )
    return session.scalars(stmt).first()


def get_extractions_for_document(
    session: Session, document_id: str
) -> list[ExtractionRecord]:
    """Return all extractions for a given document."""
    stmt = select(ExtractionRecord).where(
        ExtractionRecord.document_id == document_id
    )
    return list(session.scalars(stmt).all())


# ============================================================
# Metric CRUD
# ============================================================


def create_metric(session: Session, record: MetricRecord) -> MetricRecord:
    """Persist a MetricRecord."""
    session.add(record)
    session.flush()
    return record


def get_metrics_for_extraction(
    session: Session, extraction_id: str
) -> list[MetricRecord]:
    """Return all metrics for a given extraction."""
    stmt = select(MetricRecord).where(MetricRecord.extraction_id == extraction_id)
    return list(session.scalars(stmt).all())


# ============================================================
# Provenance CRUD
# ============================================================


def create_provenance(session: Session, record: ProvenanceRecord) -> ProvenanceRecord:
    """Persist a ProvenanceRecord."""
    session.add(record)
    session.flush()
    return record


def get_provenance_for_extraction(
    session: Session, extraction_id: str
) -> list[ProvenanceRecord]:
    """Return all provenance records for a given extraction."""
    stmt = select(ProvenanceRecord).where(
        ProvenanceRecord.extraction_id == extraction_id
    )
    return list(session.scalars(stmt).all())


# ============================================================
# Evaluation CRUD
# ============================================================


def create_evaluation(session: Session, record: EvaluationRecord) -> EvaluationRecord:
    """Persist an EvaluationRecord."""
    session.add(record)
    session.flush()
    return record


def list_evaluations(
    session: Session,
    *,
    pipeline: str | None = None,
    experiment_id: str | None = None,
) -> list[EvaluationRecord]:
    """List evaluations, optionally filtered by pipeline or experiment."""
    stmt = select(EvaluationRecord)
    if pipeline:
        stmt = stmt.where(EvaluationRecord.pipeline == pipeline)
    if experiment_id:
        stmt = stmt.where(EvaluationRecord.experiment_id == experiment_id)
    return list(session.scalars(stmt).all())


# ============================================================
# Failure CRUD
# ============================================================


def create_failure(session: Session, record: FailureRecord) -> FailureRecord:
    """Persist a FailureRecord."""
    session.add(record)
    session.flush()
    return record


def list_failures(
    session: Session,
    *,
    pipeline: str | None = None,
    failure_type: str | None = None,
    document_id: str | None = None,
) -> list[FailureRecord]:
    """List failures with optional filters."""
    stmt = select(FailureRecord)
    if pipeline:
        stmt = stmt.where(FailureRecord.pipeline == pipeline)
    if failure_type:
        stmt = stmt.where(FailureRecord.failure_type == failure_type)
    if document_id:
        stmt = stmt.where(FailureRecord.document_id == document_id)
    return list(session.scalars(stmt).all())


# ============================================================
# Experiment CRUD
# ============================================================


def create_experiment(session: Session, record: ExperimentRecord) -> ExperimentRecord:
    """Persist an ExperimentRecord."""
    session.add(record)
    session.flush()
    logger.info("Created experiment %s [%s/%s]", record.experiment_id, record.pipeline, record.dataset)
    return record


def get_experiment(session: Session, experiment_id: str) -> ExperimentRecord | None:
    """Fetch an experiment by ID."""
    stmt = select(ExperimentRecord).where(
        ExperimentRecord.experiment_id == experiment_id
    )
    return session.scalars(stmt).first()


def list_experiments(session: Session) -> list[ExperimentRecord]:
    """List all experiments ordered by start time descending."""
    stmt = select(ExperimentRecord).order_by(ExperimentRecord.started_at.desc())
    return list(session.scalars(stmt).all())

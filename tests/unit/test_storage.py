"""
Unit tests for finextract.storage.repository.

Verifies CRUD operations against the in-memory SQLite fixture from conftest.py.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

from finextract.storage.models import (
    DocumentRecord,
    ExperimentRecord,
    ExtractionRecord,
    FailureRecord,
    MetricRecord,
)
from finextract.storage.repository import (
    create_document,
    create_experiment,
    create_extraction,
    create_failure,
    create_metric,
    get_document,
    get_extraction,
    get_extractions_for_document,
    get_metrics_for_extraction,
    list_documents,
    list_experiments,
    list_failures,
    update_document_status,
)


def make_doc_id() -> str:
    return str(uuid.uuid4())


def make_extraction_id() -> str:
    return str(uuid.uuid4())


# ============================================================
# Documents
# ============================================================


class TestDocumentCRUD:
    def test_create_and_fetch(self, db_session: Session):
        doc_id = make_doc_id()
        doc = DocumentRecord(
            document_id=doc_id,
            company="Apple Inc.",
            fiscal_year=2023,
            filename="apple_2023_10k.pdf",
        )
        created = create_document(db_session, doc)
        assert created.id is not None
        assert created.document_id == doc_id

        fetched = get_document(db_session, doc_id)
        assert fetched is not None
        assert fetched.company == "Apple Inc."
        assert fetched.fiscal_year == 2023

    def test_fetch_nonexistent_returns_none(self, db_session: Session):
        result = get_document(db_session, "nonexistent-id")
        assert result is None

    def test_list_documents(self, db_session: Session):
        for i in range(3):
            create_document(
                db_session,
                DocumentRecord(
                    document_id=make_doc_id(),
                    company=f"Company{i}",
                    fiscal_year=2023,
                    filename=f"file{i}.pdf",
                ),
            )
        docs = list_documents(db_session)
        assert len(docs) == 3

    def test_update_status(self, db_session: Session):
        doc_id = make_doc_id()
        doc = DocumentRecord(
            document_id=doc_id,
            company="Microsoft",
            fiscal_year=2023,
            filename="msft.pdf",
        )
        create_document(db_session, doc)
        update_document_status(db_session, doc_id, "extracted")
        fetched = get_document(db_session, doc_id)
        assert fetched.status == "extracted"

    def test_update_status_nonexistent_raises(self, db_session: Session):
        with pytest.raises(ValueError, match="not found"):
            update_document_status(db_session, "bad-id", "extracted")


# ============================================================
# Extractions
# ============================================================


class TestExtractionCRUD:
    def _make_doc(self, session: Session) -> str:
        doc_id = make_doc_id()
        create_document(
            session,
            DocumentRecord(
                document_id=doc_id,
                company="Test Co",
                fiscal_year=2023,
                filename="test.pdf",
            ),
        )
        return doc_id

    def test_create_and_fetch(self, db_session: Session):
        doc_id = self._make_doc(db_session)
        ext_id = make_extraction_id()
        extraction = ExtractionRecord(
            extraction_id=ext_id,
            document_id=doc_id,
            pipeline="text_only",
        )
        create_extraction(db_session, extraction)

        fetched = get_extraction(db_session, ext_id)
        assert fetched is not None
        assert fetched.pipeline == "text_only"

    def test_get_extractions_for_document(self, db_session: Session):
        doc_id = self._make_doc(db_session)
        for pipeline in ["text_only", "layout_aware", "hybrid"]:
            create_extraction(
                db_session,
                ExtractionRecord(
                    extraction_id=make_extraction_id(),
                    document_id=doc_id,
                    pipeline=pipeline,
                ),
            )
        extractions = get_extractions_for_document(db_session, doc_id)
        assert len(extractions) == 3
        pipelines = {e.pipeline for e in extractions}
        assert pipelines == {"text_only", "layout_aware", "hybrid"}


# ============================================================
# Metrics
# ============================================================


class TestMetricCRUD:
    def _setup(self, session: Session) -> tuple[str, str]:
        doc_id = make_doc_id()
        ext_id = make_extraction_id()
        create_document(
            session,
            DocumentRecord(document_id=doc_id, company="C", fiscal_year=2023, filename="f.pdf"),
        )
        create_extraction(
            session,
            ExtractionRecord(extraction_id=ext_id, document_id=doc_id, pipeline="mock"),
        )
        return doc_id, ext_id

    def test_create_and_list(self, db_session: Session):
        _, ext_id = self._setup(db_session)
        for field in ["revenue", "net_income"]:
            create_metric(
                db_session,
                MetricRecord(
                    extraction_id=ext_id,
                    field_name=field,
                    value=100.0,
                    unit="million USD",
                    currency="USD",
                ),
            )
        metrics = get_metrics_for_extraction(db_session, ext_id)
        assert len(metrics) == 2
        fields = {m.field_name for m in metrics}
        assert fields == {"revenue", "net_income"}


# ============================================================
# Failures
# ============================================================


class TestFailureCRUD:
    def _setup(self, session: Session) -> tuple[str, str]:
        doc_id = make_doc_id()
        ext_id = make_extraction_id()
        create_document(
            session,
            DocumentRecord(document_id=doc_id, company="C", fiscal_year=2023, filename="f.pdf"),
        )
        create_extraction(
            session,
            ExtractionRecord(extraction_id=ext_id, document_id=doc_id, pipeline="text_only"),
        )
        return doc_id, ext_id

    def test_create_and_list_by_pipeline(self, db_session: Session):
        doc_id, ext_id = self._setup(db_session)
        failure_id = str(uuid.uuid4())
        create_failure(
            db_session,
            FailureRecord(
                failure_id=failure_id,
                extraction_id=ext_id,
                document_id=doc_id,
                field="revenue",
                failure_type="MISSING_VALUE",
                pipeline="text_only",
            ),
        )
        failures = list_failures(db_session, pipeline="text_only")
        assert len(failures) == 1
        assert failures[0].failure_type == "MISSING_VALUE"

    def test_filter_by_failure_type(self, db_session: Session):
        doc_id, ext_id = self._setup(db_session)
        for ft in ["SIGN_ERROR", "MISSING_VALUE", "SIGN_ERROR"]:
            create_failure(
                db_session,
                FailureRecord(
                    failure_id=str(uuid.uuid4()),
                    extraction_id=ext_id,
                    document_id=doc_id,
                    field="eps",
                    failure_type=ft,
                    pipeline="text_only",
                ),
            )
        sign_errors = list_failures(db_session, failure_type="SIGN_ERROR")
        assert len(sign_errors) == 2


# ============================================================
# Experiments
# ============================================================


class TestExperimentCRUD:
    def test_create_and_list(self, db_session: Session):
        exp = ExperimentRecord(
            experiment_id=str(uuid.uuid4()),
            pipeline="hybrid",
            dataset="sample",
            llm_provider="mock",
            llm_model="mock-model-v1",
        )
        create_experiment(db_session, exp)
        experiments = list_experiments(db_session)
        assert len(experiments) == 1
        assert experiments[0].pipeline == "hybrid"

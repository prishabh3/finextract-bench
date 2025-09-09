"""
Unit tests for finextract.validation.schemas.

Covers:
- FinancialMetric validation (value, currency, bbox, confidence)
- FinancialReport validation (fiscal_year, metric_fields, coverage)
- ProvenanceRecord validation
- FieldComparison auto-computed error metrics
- FailureRecord
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from finextract.validation.schemas import (
    ExtractionMethod,
    FieldComparison,
    FinancialMetric,
    FinancialReport,
    ProvenanceRecord,
    ValidationStatus,
)

# ============================================================
# FinancialMetric
# ============================================================


class TestFinancialMetric:
    def test_minimal_valid(self):
        m = FinancialMetric(value=100.0)
        assert m.value == 100.0
        assert m.unit == "USD"
        assert m.currency is None
        assert m.bbox is None
        assert m.confidence is None

    def test_full_valid(self):
        m = FinancialMetric(
            value=416161.0,
            original_value="$416,161",
            unit="million USD",
            currency="USD",
            page=31,
            source_text="Total net sales 416,161",
            bbox=[100.0, 200.0, 450.0, 250.0],
            confidence=0.96,
            extraction_method=ExtractionMethod.HYBRID,
        )
        assert m.value == pytest.approx(416161.0)
        assert m.currency == "USD"
        assert m.page == 31
        assert m.confidence == pytest.approx(0.96)

    def test_nan_value_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            FinancialMetric(value=float("nan"))

    def test_inf_value_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            FinancialMetric(value=float("inf"))

    def test_invalid_currency_format_raises(self):
        """Currency must be exactly 3 uppercase letters."""
        with pytest.raises(ValidationError, match="ISO 4217"):
            FinancialMetric(value=100.0, currency="DOLLARS")

    def test_lowercase_currency_normalized(self):
        """Currency should be uppercased by validator."""
        m = FinancialMetric(value=100.0, currency="usd")
        assert m.currency == "USD"

    def test_invalid_bbox_length(self):
        with pytest.raises(ValidationError):
            FinancialMetric(value=100.0, bbox=[10.0, 20.0, 30.0])  # Only 3 values

    def test_invalid_bbox_coordinates(self):
        """x0 >= x1 should fail."""
        with pytest.raises(ValidationError, match="x0"):
            FinancialMetric(value=100.0, bbox=[100.0, 10.0, 50.0, 50.0])

    def test_invalid_bbox_y(self):
        """y0 >= y1 should fail."""
        with pytest.raises(ValidationError, match="y0"):
            FinancialMetric(value=100.0, bbox=[10.0, 100.0, 50.0, 50.0])

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValidationError, match="confidence"):
            FinancialMetric(value=100.0, confidence=1.5)

    def test_confidence_below_0_raises(self):
        with pytest.raises(ValidationError, match="confidence"):
            FinancialMetric(value=100.0, confidence=-0.1)

    def test_confidence_boundary_values(self):
        """0.0 and 1.0 are valid confidence values."""
        m_low = FinancialMetric(value=100.0, confidence=0.0)
        m_high = FinancialMetric(value=100.0, confidence=1.0)
        assert m_low.confidence == 0.0
        assert m_high.confidence == 1.0

    def test_page_must_be_positive(self):
        with pytest.raises(ValidationError):
            FinancialMetric(value=100.0, page=0)

    def test_negative_value_valid(self):
        """Negative values (losses) are valid."""
        m = FinancialMetric(value=-12500.0)
        assert m.value == pytest.approx(-12500.0)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            FinancialMetric(value=100.0, unknown_field="bad")


# ============================================================
# FinancialReport
# ============================================================


class TestFinancialReport:
    def test_minimal_valid(self):
        r = FinancialReport(company="Apple Inc.", fiscal_year=2023)
        assert r.company == "Apple Inc."
        assert r.fiscal_year == 2023
        assert r.revenue is None
        assert r.validation_status == ValidationStatus.VALID

    def test_string_fiscal_year_coerced(self):
        r = FinancialReport(company="Apple Inc.", fiscal_year="2023")
        assert r.fiscal_year == 2023

    def test_invalid_fiscal_year_string(self):
        with pytest.raises(ValidationError):
            FinancialReport(company="Apple Inc.", fiscal_year="twenty-twenty")

    def test_fiscal_year_too_old(self):
        with pytest.raises(ValidationError):
            FinancialReport(company="Apple Inc.", fiscal_year=1800)

    def test_fiscal_year_future(self):
        with pytest.raises(ValidationError):
            FinancialReport(company="Apple Inc.", fiscal_year=2200)

    def test_with_metrics(self):
        r = FinancialReport(
            company="Apple Inc.",
            fiscal_year=2023,
            revenue=FinancialMetric(value=383285.0, unit="million USD", currency="USD"),
            net_income=FinancialMetric(value=96995.0, unit="million USD", currency="USD"),
        )
        assert r.revenue is not None
        assert r.revenue.value == pytest.approx(383285.0)
        assert r.net_income is not None

    def test_metric_fields_returns_all(self):
        r = FinancialReport(company="Test", fiscal_year=2023)
        fields = r.metric_fields()
        expected = {
            "revenue",
            "net_income",
            "operating_income",
            "total_assets",
            "total_liabilities",
            "cash_and_equivalents",
            "eps",
        }
        assert set(fields.keys()) == expected

    def test_extraction_coverage_zero(self):
        r = FinancialReport(company="Test", fiscal_year=2023)
        assert r.extraction_coverage() == pytest.approx(0.0)

    def test_extraction_coverage_partial(self):
        r = FinancialReport(
            company="Test",
            fiscal_year=2023,
            revenue=FinancialMetric(value=100.0),
            net_income=FinancialMetric(value=50.0),
        )
        # 2 out of 7 fields
        assert r.extraction_coverage() == pytest.approx(2 / 7)

    def test_extraction_coverage_full(self):
        metric = FinancialMetric(value=100.0)
        r = FinancialReport(
            company="Test",
            fiscal_year=2023,
            revenue=metric,
            net_income=metric,
            operating_income=metric,
            total_assets=metric,
            total_liabilities=metric,
            cash_and_equivalents=metric,
            eps=metric,
        )
        assert r.extraction_coverage() == pytest.approx(1.0)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            FinancialReport(company="Test", fiscal_year=2023, unknown_field=123)


# ============================================================
# ProvenanceRecord
# ============================================================


class TestProvenanceRecord:
    def test_minimal_valid(self):
        p = ProvenanceRecord(
            document_id="doc-001",
            company="Apple Inc.",
            fiscal_year=2023,
            field_name="revenue",
        )
        assert p.document_id == "doc-001"
        assert p.llm_provider is None
        assert p.input_tokens is None

    def test_all_fields(self):
        p = ProvenanceRecord(
            document_id="doc-001",
            company="Apple Inc.",
            fiscal_year=2023,
            field_name="revenue",
            page=31,
            source_text="Total net sales 416,161",
            extraction_method=ExtractionMethod.HYBRID,
            parser_name="docling",
            parser_version="2.5.0",
            llm_provider="openai",
            llm_model="gpt-4o",
            confidence=0.96,
            input_tokens=1500,
            output_tokens=200,
            estimated_cost_usd=0.0042,
            processing_time_ms=3200.0,
        )
        assert p.input_tokens == 1500
        assert p.llm_model == "gpt-4o"

    def test_negative_input_tokens_raises(self):
        with pytest.raises(ValidationError):
            ProvenanceRecord(
                document_id="d",
                company="C",
                fiscal_year=2023,
                field_name="revenue",
                input_tokens=-1,
            )


# ============================================================
# FieldComparison — auto-computed errors
# ============================================================


class TestFieldComparison:
    def test_exact_match(self):
        fc = FieldComparison(
            field_name="revenue",
            predicted_value=100.0,
            ground_truth_value=100.0,
        )
        assert fc.exact_match is True
        assert fc.absolute_error == pytest.approx(0.0)
        assert fc.relative_error == pytest.approx(0.0)
        assert fc.within_05pct is True
        assert fc.within_1pct is True
        assert fc.within_5pct is True

    def test_within_05pct(self):
        fc = FieldComparison(
            field_name="revenue",
            predicted_value=100.4,
            ground_truth_value=100.0,
        )
        assert fc.exact_match is False
        assert fc.within_05pct is True
        assert fc.relative_error == pytest.approx(0.004)

    def test_within_1pct(self):
        fc = FieldComparison(
            field_name="revenue",
            predicted_value=100.8,
            ground_truth_value=100.0,
        )
        assert fc.within_05pct is False
        assert fc.within_1pct is True

    def test_within_5pct(self):
        fc = FieldComparison(
            field_name="revenue",
            predicted_value=104.0,
            ground_truth_value=100.0,
        )
        assert fc.within_1pct is False
        assert fc.within_5pct is True

    def test_outside_all_tolerances(self):
        fc = FieldComparison(
            field_name="revenue",
            predicted_value=110.0,
            ground_truth_value=100.0,
        )
        assert fc.within_5pct is False
        assert fc.relative_error == pytest.approx(0.10)

    def test_missing_predicted(self):
        fc = FieldComparison(
            field_name="revenue",
            predicted_value=None,
            ground_truth_value=100.0,
        )
        assert fc.absolute_error is None
        assert fc.exact_match is None

    def test_missing_ground_truth(self):
        fc = FieldComparison(
            field_name="revenue",
            predicted_value=100.0,
            ground_truth_value=None,
        )
        assert fc.absolute_error is None

    def test_zero_ground_truth(self):
        """Relative error is undefined when ground truth is zero."""
        fc = FieldComparison(
            field_name="eps",
            predicted_value=0.0,
            ground_truth_value=0.0,
        )
        # Exact match should be True even for zero
        assert fc.exact_match is True

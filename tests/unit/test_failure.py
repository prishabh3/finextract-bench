"""
Unit tests for finextract.evaluation.failure.

Covers:
- classify_failure: MISSING_VALUE, SIGN_ERROR, UNIT_NORMALIZATION, NEEDS_REVIEW
- compute_failure_severity
- summarize_failures
"""

from __future__ import annotations

from finextract.evaluation.failure import (
    MISSING_VALUE_TYPE,
    NEEDS_REVIEW_TYPE,
    SIGN_ERROR_TYPE,
    UNIT_NORMALIZATION_TYPE,
    VALIDATION_ERROR_TYPE,
    classify_failure,
    compute_failure_severity,
    summarize_failures,
)

# ============================================================
# classify_failure
# ============================================================


class TestClassifyFailure:
    def test_missing_predicted(self):
        ft, cause, notes = classify_failure(
            field="revenue",
            predicted=None,
            ground_truth=100.0,
        )
        assert ft == MISSING_VALUE_TYPE
        assert cause is not None

    def test_missing_gt_needs_review(self):
        ft, cause, notes = classify_failure(
            field="revenue",
            predicted=100.0,
            ground_truth=None,
        )
        assert ft == NEEDS_REVIEW_TYPE

    def test_validation_error(self):
        ft, cause, notes = classify_failure(
            field="revenue",
            predicted=100.0,
            ground_truth=100.0,
            is_validation_error=True,
        )
        assert ft == VALIDATION_ERROR_TYPE

    def test_sign_error_positive_to_negative(self):
        ft, cause, notes = classify_failure(
            field="net_income",
            predicted=-50_000.0,
            ground_truth=50_000.0,
        )
        assert ft == SIGN_ERROR_TYPE
        assert cause is not None

    def test_sign_error_negative_to_positive(self):
        ft, cause, notes = classify_failure(
            field="net_income",
            predicted=50_000.0,
            ground_truth=-50_000.0,
        )
        assert ft == SIGN_ERROR_TYPE

    def test_unit_normalization_thousands_to_millions(self):
        """Predicted is 1000x too large (forgot millions → took raw thousands)."""
        ft, cause, notes = classify_failure(
            field="revenue",
            predicted=416_161_000.0,   # value as plain dollars
            ground_truth=416_161.0,    # value in millions
        )
        assert ft == UNIT_NORMALIZATION_TYPE
        assert cause is not None

    def test_unit_normalization_millions_to_billions(self):
        """Predicted is 1000x too large (millions vs billions confusion)."""
        ft, cause, notes = classify_failure(
            field="total_assets",
            predicted=352_755e3,   # thinks it's millions, actually thousands
            ground_truth=352_755.0,
        )
        assert ft == UNIT_NORMALIZATION_TYPE

    def test_unit_normalization_inverse(self):
        """Predicted is 1000x too small."""
        ft, cause, notes = classify_failure(
            field="revenue",
            predicted=416.161,
            ground_truth=416_161.0,
        )
        assert ft == UNIT_NORMALIZATION_TYPE

    def test_ambiguous_error_needs_review(self):
        """A 15% error doesn't match any specific failure pattern."""
        ft, cause, notes = classify_failure(
            field="eps",
            predicted=7.0,
            ground_truth=6.0,
        )
        assert ft == NEEDS_REVIEW_TYPE

    def test_exact_match_sign_error_not_triggered(self):
        """Ratio = 1.0 should not trigger sign error."""
        ft, cause, notes = classify_failure(
            field="revenue",
            predicted=100.0,
            ground_truth=100.0,
        )
        # Should be NEEDS_REVIEW (no error, so not classified as failure)
        # In practice, classify_failure is only called on actual failures, but
        # it should at least not trigger a wrong classification.
        assert ft in (NEEDS_REVIEW_TYPE, SIGN_ERROR_TYPE) is False or ft == NEEDS_REVIEW_TYPE


# ============================================================
# compute_failure_severity
# ============================================================


class TestComputeFailureSeverity:
    def test_missing_value_medium(self):
        severity = compute_failure_severity(MISSING_VALUE_TYPE, None)
        assert severity == "medium"

    def test_validation_error_high(self):
        severity = compute_failure_severity(VALIDATION_ERROR_TYPE, None)
        assert severity == "high"

    def test_sign_error_high(self):
        severity = compute_failure_severity(SIGN_ERROR_TYPE, 2.0)
        assert severity == "high"

    def test_unit_normalization_critical(self):
        severity = compute_failure_severity(UNIT_NORMALIZATION_TYPE, 999.0)
        assert severity == "critical"

    def test_small_needs_review_low(self):
        severity = compute_failure_severity(NEEDS_REVIEW_TYPE, 0.005)
        assert severity == "low"

    def test_large_needs_review_critical(self):
        severity = compute_failure_severity(NEEDS_REVIEW_TYPE, 0.50)
        assert severity == "critical"

    def test_no_relative_error_medium(self):
        severity = compute_failure_severity(NEEDS_REVIEW_TYPE, None)
        assert severity == "medium"


# ============================================================
# summarize_failures
# ============================================================


class TestSummarizeFailures:
    def test_empty(self):
        result = summarize_failures([])
        assert result == {}

    def test_single_type(self):
        failures = [
            {"failure_type": "MISSING_VALUE"},
            {"failure_type": "MISSING_VALUE"},
            {"failure_type": "MISSING_VALUE"},
        ]
        result = summarize_failures(failures)
        assert result["MISSING_VALUE"] == 3

    def test_multiple_types(self):
        failures = [
            {"failure_type": "MISSING_VALUE"},
            {"failure_type": "SIGN_ERROR"},
            {"failure_type": "MISSING_VALUE"},
            {"failure_type": "UNIT_NORMALIZATION"},
        ]
        result = summarize_failures(failures)
        assert result["MISSING_VALUE"] == 2
        assert result["SIGN_ERROR"] == 1
        assert result["UNIT_NORMALIZATION"] == 1

    def test_sorted_by_count_descending(self):
        failures = [
            {"failure_type": "SIGN_ERROR"},
            {"failure_type": "MISSING_VALUE"},
            {"failure_type": "MISSING_VALUE"},
        ]
        result = summarize_failures(failures)
        keys = list(result.keys())
        assert keys[0] == "MISSING_VALUE"
        assert keys[1] == "SIGN_ERROR"

    def test_missing_failure_type_defaults_to_other(self):
        failures = [{"no_type_key": True}]
        result = summarize_failures(failures)
        assert result.get("OTHER", 0) == 1

"""
Unit tests for finextract.evaluation.metrics.

Covers:
- compute_field_result: error calculations, tolerance levels, edge cases
- compute_latency_stats: mean, median, p95, min/max, empty input
- compute_pipeline_metrics: aggregation, coverage, accuracy, latency, cost
"""

from __future__ import annotations

import pytest

from finextract.evaluation.metrics import (
    FieldResult,
    compute_field_result,
    compute_latency_stats,
    compute_pipeline_metrics,
)

# ============================================================
# compute_field_result
# ============================================================


class TestComputeFieldResult:
    def test_exact_match(self):
        r = compute_field_result("revenue", 100.0, 100.0)
        assert r.exact_match is True
        assert r.absolute_error == pytest.approx(0.0)
        assert r.relative_error == pytest.approx(0.0)
        assert r.within_05pct is True
        assert r.within_1pct is True
        assert r.within_5pct is True

    def test_small_relative_error(self):
        r = compute_field_result("revenue", 100.4, 100.0)
        assert r.exact_match is False
        assert r.relative_error == pytest.approx(0.004)
        assert r.within_05pct is True
        assert r.within_1pct is True
        assert r.within_5pct is True

    def test_moderate_error(self):
        r = compute_field_result("revenue", 101.5, 100.0)
        assert r.within_05pct is False
        assert r.within_1pct is False
        assert r.within_5pct is True

    def test_large_error(self):
        r = compute_field_result("revenue", 120.0, 100.0)
        assert r.within_5pct is False
        assert r.relative_error == pytest.approx(0.20)

    def test_zero_gt_no_relative_error(self):
        r = compute_field_result("field", 10.0, 0.0)
        assert r.absolute_error == pytest.approx(10.0)
        assert r.relative_error is None
        assert r.within_05pct is None
        assert "undefined" in r.notes.lower()

    def test_missing_predicted(self):
        r = compute_field_result("revenue", None, 100.0)
        assert r.absolute_error is None
        assert r.exact_match is None
        assert "missing" in r.notes.lower()

    def test_missing_ground_truth(self):
        r = compute_field_result("revenue", 100.0, None)
        assert r.absolute_error is None

    def test_negative_values(self):
        r = compute_field_result("net_income", -50.0, -50.0)
        assert r.exact_match is True
        assert r.absolute_error == pytest.approx(0.0)

    def test_unit_preserved(self):
        r = compute_field_result(
            "revenue", 100.0, 100.0,
            predicted_unit="million USD",
            ground_truth_unit="million USD",
        )
        assert r.predicted_unit == "million USD"
        assert r.ground_truth_unit == "million USD"


# ============================================================
# compute_latency_stats
# ============================================================


class TestComputeLatencyStats:
    def test_empty(self):
        stats = compute_latency_stats([])
        assert stats["mean"] is None
        assert stats["median"] is None
        assert stats["p95"] is None
        assert stats["min"] is None
        assert stats["max"] is None

    def test_single_value(self):
        stats = compute_latency_stats([100.0])
        assert stats["mean"] == pytest.approx(100.0)
        assert stats["median"] == pytest.approx(100.0)
        assert stats["p95"] == pytest.approx(100.0)
        assert stats["min"] == pytest.approx(100.0)
        assert stats["max"] == pytest.approx(100.0)

    def test_multiple_values(self):
        latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
        stats = compute_latency_stats(latencies)
        assert stats["mean"] == pytest.approx(300.0)
        assert stats["median"] == pytest.approx(300.0)
        assert stats["min"] == pytest.approx(100.0)
        assert stats["max"] == pytest.approx(500.0)

    def test_p95_is_at_correct_percentile(self):
        # 20 values; p95 should be in the top few
        latencies = list(range(1, 21))  # 1..20
        stats = compute_latency_stats(latencies)
        # p95 index = int(0.95 * 20 + 0.5) = 19 → value = 19+1 = 20, but 0-indexed so value[19]=20
        # Actually with 0-indexed list [1..20], index 19 = value 20
        assert stats["p95"] == pytest.approx(20.0)
        assert stats["max"] == pytest.approx(20.0)

    def test_unordered_input(self):
        """Input doesn't need to be sorted."""
        stats = compute_latency_stats([500.0, 100.0, 300.0])
        assert stats["min"] == pytest.approx(100.0)
        assert stats["max"] == pytest.approx(500.0)


# ============================================================
# compute_pipeline_metrics
# ============================================================


class TestComputePipelineMetrics:
    def _make_results(self, cases: list[tuple]) -> list[FieldResult]:
        """
        Helper: list of (field, predicted, gt) tuples → list of FieldResult.
        """
        results = []
        for field, pred, gt in cases:
            r = compute_field_result(field, pred, gt)
            results.append(r)
        return results

    def test_all_correct(self):
        results = self._make_results([
            ("revenue", 100.0, 100.0),
            ("net_income", 50.0, 50.0),
        ])
        m = compute_pipeline_metrics("text_only", results, document_count=1)
        assert m.exact_accuracy == pytest.approx(1.0)
        assert m.extraction_coverage == pytest.approx(1.0)
        assert m.missing_fields == 0

    def test_all_missing(self):
        results = self._make_results([
            ("revenue", None, 100.0),
            ("net_income", None, 50.0),
        ])
        m = compute_pipeline_metrics("text_only", results, document_count=1)
        assert m.extraction_coverage == pytest.approx(0.0)
        assert m.missing_fields == 2
        # Accuracy is undefined when nothing is comparable
        assert m.exact_accuracy == pytest.approx(0.0)

    def test_partial_extraction(self):
        results = self._make_results([
            ("revenue", 100.0, 100.0),     # extracted + correct
            ("net_income", 60.0, 50.0),    # extracted + wrong (20% error)
            ("operating_income", None, 30.0),  # missing
        ])
        m = compute_pipeline_metrics("text_only", results, document_count=1)
        assert m.extraction_coverage == pytest.approx(2 / 3)
        assert m.extracted_fields == 2
        assert m.missing_fields == 1
        # Among comparable: 1 exact match out of 2
        assert m.exact_accuracy == pytest.approx(0.5)

    def test_latency_stats_populated(self):
        results = self._make_results([("revenue", 100.0, 100.0)])
        latencies = [1000.0, 2000.0, 3000.0]
        m = compute_pipeline_metrics(
            "hybrid", results, latencies_ms=latencies, document_count=3
        )
        assert m.mean_latency_ms == pytest.approx(2000.0)
        assert m.median_latency_ms == pytest.approx(2000.0)
        assert m.min_latency_ms == pytest.approx(1000.0)
        assert m.max_latency_ms == pytest.approx(3000.0)

    def test_cost_metrics(self):
        results = self._make_results([
            ("revenue", 100.0, 100.0),
            ("net_income", 50.0, 50.0),
        ])
        m = compute_pipeline_metrics(
            "hybrid",
            results,
            total_cost_usd=0.10,
            document_count=2,
        )
        assert m.total_cost_usd == pytest.approx(0.10)
        assert m.cost_per_document_usd == pytest.approx(0.05)
        # cost / total_fields * 100 = 0.10 / 2 * 100 = 5.0
        assert m.cost_per_100_fields_usd == pytest.approx(5.0)

    def test_validation_failure_rate(self):
        results = []
        m = compute_pipeline_metrics(
            "text_only",
            results,
            validation_failures=3,
            total_attempted=10,
        )
        assert m.validation_failure_rate == pytest.approx(0.3)

    def test_per_field_accuracy(self):
        results = [
            compute_field_result("revenue", 100.0, 100.0),
            compute_field_result("revenue", 101.0, 100.0),  # 1% error = within_1pct border
            compute_field_result("net_income", 50.0, 50.0),
        ]
        m = compute_pipeline_metrics("hybrid", results, document_count=1)
        # revenue: 1 of 2 within 1% (101/100 = 1% — borderline, within_1pct=True at exactly 0.01)
        assert "revenue" in m.per_field_accuracy
        assert "net_income" in m.per_field_accuracy
        assert m.per_field_accuracy["net_income"] == pytest.approx(1.0)

    def test_empty_results(self):
        m = compute_pipeline_metrics("text_only", [], document_count=0)
        assert m.total_fields == 0
        assert m.extraction_coverage == pytest.approx(0.0)
        assert m.exact_accuracy == pytest.approx(0.0)

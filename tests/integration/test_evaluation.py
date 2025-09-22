"""
FinExtract-Bench: Phase 4 Evaluation integration tests.

Tests cover:
- GroundTruthLoader (loading CSV, getting by company/year, default mock record)
- Evaluation Harness (evaluate_report, field comparisons, failure classification)
- PipelineMetrics aggregation
- Cost estimation via evaluate_experiment and TokenUsage
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finextract.data.sample_pdf import SAMPLE_DATA
from finextract.evaluation.failure import MISSING_VALUE_TYPE
from finextract.evaluation.ground_truth import (
    GroundTruthLoader,
    GroundTruthNotFoundError,
    GroundTruthRecord,
)
from finextract.evaluation.harness import estimate_run_cost, evaluate_experiment, evaluate_report
from finextract.extraction.providers.base import LLMResponse
from finextract.validation.schemas import ExtractionMethod, FinancialMetric, FinancialReport


@pytest.fixture
def gt_loader(tmp_path: Path) -> GroundTruthLoader:
    # Create a temporary CSV
    csv_file = tmp_path / "test_gt.csv"
    csv_content = (
        "company,fiscal_year,revenue,net_income,operating_income,total_assets,total_liabilities,cash_and_equivalents,eps,currency,unit,source\n"
        "TestCorp,2022,100,20,30,500,200,50,1.5,USD,million USD,Test Source\n"
        "AnotherCorp,2021,,,15,,,,,,,\n"
    )
    csv_file.write_text(csv_content)
    return GroundTruthLoader(csv_path=csv_file)


class TestGroundTruthLoader:
    def test_load_from_csv(self, gt_loader: GroundTruthLoader):
        record = gt_loader.get("TestCorp", 2022)
        assert record.revenue == 100.0
        assert record.net_income == 20.0
        assert record.eps == 1.5
        assert record.unit == "million USD"

    def test_case_insensitive_lookup(self, gt_loader: GroundTruthLoader):
        record = gt_loader.get("testcorp.", 2022)
        assert record.revenue == 100.0

    def test_synthetic_record_injected(self, gt_loader: GroundTruthLoader):
        record = gt_loader.get("TechCorp Inc.", 2023)
        assert record.revenue == SAMPLE_DATA["revenue"]
        assert record.net_income == SAMPLE_DATA["net_income"]

    def test_missing_fields_are_none(self, gt_loader: GroundTruthLoader):
        record = gt_loader.get("AnotherCorp", 2021)
        assert record.operating_income == 15.0
        assert record.revenue is None

    def test_not_found_raises(self, gt_loader: GroundTruthLoader):
        with pytest.raises(GroundTruthNotFoundError):
            gt_loader.get("NonExistent", 2022)


class TestEvaluationHarness:
    @pytest.fixture
    def sample_gt(self) -> GroundTruthRecord:
        return GroundTruthRecord(
            company="EvalCorp",
            fiscal_year=2024,
            revenue=100.0,
            net_income=20.0,
            eps=1.5,
            unit="million USD",
        )

    def test_evaluate_report_perfect_match(self, sample_gt: GroundTruthRecord):
        report = FinancialReport(
            company="EvalCorp",
            fiscal_year=2024,
            extraction_method=ExtractionMethod.TEXT_ONLY,
            revenue=FinancialMetric(value=100.0, unit="million USD"),
            net_income=FinancialMetric(value=20.0, unit="million USD"),
            eps=FinancialMetric(value=1.5, unit="USD"),
        )

        field_results, failures = evaluate_report(report, sample_gt, pipeline="text_only")

        # 7 total metric fields are always checked
        assert len(field_results) == 7

        rev_res = next(f for f in field_results if f.field_name == "revenue")
        assert rev_res.exact_match is True
        assert rev_res.absolute_error == 0.0

        # No failures for the fields that have ground truth (revenue, net_income, eps)
        assert len(failures) == 0

    def test_evaluate_report_with_failures(self, sample_gt: GroundTruthRecord):
        report = FinancialReport(
            company="EvalCorp",
            fiscal_year=2024,
            extraction_method=ExtractionMethod.TEXT_ONLY,
            revenue=FinancialMetric(value=100.0, unit="million USD"),  # Match
            net_income=FinancialMetric(value=20000.0, unit="USD"), # Wrong unit (scaled)
            # eps missing entirely
        )

        field_results, failures = evaluate_report(report, sample_gt, pipeline="text_only")

        # Net income scaling error and EPS missing
        assert len(failures) == 2

        eps_fail = next(f for f in failures if f["field"] == "eps")
        assert eps_fail["failure_type"] == MISSING_VALUE_TYPE

        ni_fail = next(f for f in failures if f["field"] == "net_income")
        # Since expected was 20, predicted 20000 -> 1000x off
        assert ni_fail["failure_type"] == "UNIT_NORMALIZATION"

    def test_evaluate_experiment_aggregation(self, sample_gt: GroundTruthRecord):
        r1 = FinancialReport(
            company="EvalCorp", fiscal_year=2024,
            revenue=FinancialMetric(value=100.0), # correct
        )
        r2 = FinancialReport(
            company="EvalCorp", fiscal_year=2024,
            revenue=FinancialMetric(value=110.0), # 10% error
        )

        pipeline_reports = {
            "p1": [(r1, sample_gt), (r2, sample_gt)]
        }
        latencies = {"p1": [1000.0, 1200.0]}
        costs = {"p1": 0.05}

        results = evaluate_experiment(pipeline_reports, latencies_ms=latencies, costs_usd=costs)

        assert "p1" in results
        metrics = results["p1"]
        assert metrics.document_count == 2
        # One exact match, one >5% error
        assert metrics.exact_accuracy == 0.5
        assert metrics.mean_latency_ms == 1100.0
        assert metrics.total_cost_usd == 0.05


class TestCostEstimation:
    def test_estimate_run_cost(self):
        from finextract.evaluation.cost import PricingRegistry

        class MockRegistry(PricingRegistry):
            def __init__(self):
                self._pricing = {
                    "gpt-4o-mini": {
                        "input_cost_per_1m_tokens": 0.15,
                        "output_cost_per_1m_tokens": 0.60
                    }
                }

        responses = [
            LLMResponse(text="", provider="openai", model="gpt-4o-mini", input_tokens=1000, output_tokens=100),
            LLMResponse(text="", provider="openai", model="gpt-4o-mini", input_tokens=2000, output_tokens=200),
        ]

        cost = estimate_run_cost(responses, registry=MockRegistry())
        # gpt-4o-mini: 0.15/1M input, 0.60/1M output
        # run 1: 1000 * 0.15/1M + 100 * 0.60/1M = 0.00015 + 0.00006 = 0.00021
        # run 2: 2000 * 0.15/1M + 200 * 0.60/1M = 0.00030 + 0.00012 = 0.00042
        # total: 0.00063
        assert cost == pytest.approx(0.00063)

    def test_estimate_run_cost_empty(self):
        assert estimate_run_cost([]) is None
        assert estimate_run_cost([LLMResponse(text="", provider="mock", model="m")]) is None

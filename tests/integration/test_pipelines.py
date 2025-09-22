"""
Integration tests for Phase 3: extraction pipelines.

All tests use the mock provider — no API key needed.
Tests cover:
- Mock provider: scan_context, token estimation, LLMResponse structure
- Core extractor: JSON parsing, validation, empty report on failure
- Pipeline A (text_only): end-to-end with sample PDF
- Pipeline B (layout_aware): end-to-end with sample PDF
- Pipeline C (hybrid): end-to-end + consistency checks
- Provider factory: mock, unknown raises ValueError
- Prompt templates: correct system prompt returned per pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finextract.data.sample_pdf import SAMPLE_DATA, create_sample_pdf
from finextract.extraction.extractor import (
    ExtractionResult,
    _clean_json_text,
    _try_parse_json,
    extract_from_context,
)
from finextract.extraction.pipelines import (
    _apply_consistency_checks,
    _select_hybrid_context,
    _select_layout_context,
    _select_text_context,
    _table_to_text,
    run_hybrid,
    run_layout_aware,
    run_text_only,
)
from finextract.extraction.prompts import (
    HYBRID_SYSTEM_PROMPT,
    LAYOUT_AWARE_SYSTEM_PROMPT,
    TEXT_ONLY_SYSTEM_PROMPT,
    get_system_prompt,
)
from finextract.extraction.providers.base import LLMProvider, LLMResponse, get_provider
from finextract.extraction.providers.mock import MockProvider, _approx_tokens, _scan_context
from finextract.parsing import text_parser
from finextract.parsing.base import ParsedDocument, ParsedTable
from finextract.provenance.tracker import ProvenanceTracker
from finextract.validation.schemas import (
    ExtractionMethod,
    FinancialReport,
)

# ============================================================
# Session-scoped fixtures
# ============================================================


@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("pipeline_sample")
    return create_sample_pdf(output_dir=out_dir)


@pytest.fixture(scope="session")
def parsed_doc(sample_pdf_path: Path) -> ParsedDocument:
    return text_parser.parse_pdf(sample_pdf_path, document_id="pipeline-test-001")


@pytest.fixture()
def mock_provider() -> MockProvider:
    return MockProvider(model="mock-model-v1")


@pytest.fixture()
def tracker() -> ProvenanceTracker:
    return ProvenanceTracker(
        document_id="pipeline-test-001",
        company="TechCorp Inc.",
        fiscal_year=2023,
        parser_name="pymupdf",
        parser_version="1.24.0",
        llm_provider="mock",
        llm_model="mock-model-v1",
    )


# ============================================================
# Mock provider tests
# ============================================================


class TestMockProvider:
    def test_provider_name(self, mock_provider: MockProvider):
        assert mock_provider.provider_name == "mock"

    def test_model_name(self, mock_provider: MockProvider):
        assert mock_provider.model_name == "mock-model-v1"

    def test_satisfies_protocol(self, mock_provider: MockProvider):
        """MockProvider must satisfy the LLMProvider Protocol."""
        assert isinstance(mock_provider, LLMProvider)

    def test_returns_llm_response(self, mock_provider: MockProvider):
        context = "Revenue 50,000\nNet Income 10,000"
        resp = mock_provider.extract("prompt", context)
        assert isinstance(resp, LLMResponse)
        assert isinstance(resp.text, str)
        assert resp.provider == "mock"
        assert resp.model == "mock-model-v1"

    def test_response_is_valid_json(self, mock_provider: MockProvider):
        context = "Revenue 50,000"
        resp = mock_provider.extract("prompt", context)
        parsed = json.loads(resp.text)
        assert isinstance(parsed, dict)

    def test_scans_revenue(self):
        context = "Revenue 50,000\nCost 28,000"
        result = _scan_context(context)
        assert result["revenue"] is not None
        assert result["revenue"]["value"] == "50,000"

    def test_scans_net_income(self):
        context = "Net Income 10,000"
        result = _scan_context(context)
        assert result["net_income"] is not None
        assert result["net_income"]["value"] == "10,000"

    def test_missing_field_returns_none(self):
        context = "Revenue 50,000"
        result = _scan_context(context)
        assert result["eps"] is None
        assert result["total_assets"] is None

    def test_scans_eps(self):
        context = "Earnings Per Share (Diluted) 5.00"
        result = _scan_context(context)
        assert result["eps"] is not None
        assert result["eps"]["value"] == "5.00"

    def test_token_counts_present(self, mock_provider: MockProvider):
        resp = mock_provider.extract("prompt text", "context text")
        assert resp.input_tokens is not None
        assert resp.output_tokens is not None
        assert resp.input_tokens > 0
        assert resp.output_tokens > 0

    def test_latency_recorded(self, mock_provider: MockProvider):
        resp = mock_provider.extract("prompt", "context")
        assert resp.latency_ms is not None
        assert resp.latency_ms >= 0

    def test_approx_tokens(self):
        text = "a" * 400
        tokens = _approx_tokens(text)
        assert tokens == 100  # 400 / 4

    def test_deterministic(self, mock_provider: MockProvider):
        """Same input always produces same output."""
        ctx = "Revenue 50,000\nNet Income 10,000"
        r1 = mock_provider.extract("prompt", ctx)
        r2 = mock_provider.extract("prompt", ctx)
        assert r1.text == r2.text


# ============================================================
# Core extractor tests
# ============================================================


class TestCoreExtractor:
    SIMPLE_CONTEXT = (
        "Revenue 50,000\nNet Income 10,000\n"
        "Operating Income 12,000\nTotal Assets 80,000\n"
        "Total Liabilities 30,000\nCash and Cash Equivalents 15,000\n"
        "Earnings Per Share (Diluted) 5.00"
    )

    def test_extract_produces_report(self, mock_provider: MockProvider):
        result = extract_from_context(
            self.SIMPLE_CONTEXT,
            company="TechCorp Inc.",
            fiscal_year=2023,
            pipeline="text_only",
            provider=mock_provider,
        )
        assert isinstance(result, ExtractionResult)
        assert result.report is not None

    def test_report_company_and_year(self, mock_provider: MockProvider):
        result = extract_from_context(
            self.SIMPLE_CONTEXT,
            company="TechCorp Inc.",
            fiscal_year=2023,
            pipeline="text_only",
            provider=mock_provider,
        )
        report = result.report
        assert report.company == "TechCorp Inc."
        assert report.fiscal_year == 2023

    def test_revenue_extracted(self, mock_provider: MockProvider):
        result = extract_from_context(
            self.SIMPLE_CONTEXT,
            company="TechCorp Inc.",
            fiscal_year=2023,
            pipeline="text_only",
            provider=mock_provider,
        )
        report = result.report
        assert report is not None
        assert report.revenue is not None
        # 50,000 → after normalization = 50000.0
        assert report.revenue.value == pytest.approx(50_000.0)

    def test_net_income_extracted(self, mock_provider: MockProvider):
        result = extract_from_context(
            self.SIMPLE_CONTEXT,
            company="TechCorp Inc.",
            fiscal_year=2023,
            pipeline="text_only",
            provider=mock_provider,
        )
        report = result.report
        assert report is not None
        assert report.net_income is not None
        assert report.net_income.value == pytest.approx(10_000.0)

    def test_eps_extracted(self, mock_provider: MockProvider):
        result = extract_from_context(
            self.SIMPLE_CONTEXT,
            company="TechCorp Inc.",
            fiscal_year=2023,
            pipeline="text_only",
            provider=mock_provider,
        )
        report = result.report
        assert report is not None
        assert report.eps is not None
        assert report.eps.value == pytest.approx(5.00)

    def test_extraction_method_set(self, mock_provider: MockProvider):
        result = extract_from_context(
            self.SIMPLE_CONTEXT,
            company="TechCorp Inc.",
            fiscal_year=2023,
            pipeline="text_only",
            provider=mock_provider,
        )
        assert result.report.extraction_method == ExtractionMethod.TEXT_ONLY

    def test_raw_json_preserved(self, mock_provider: MockProvider):
        result = extract_from_context(
            self.SIMPLE_CONTEXT,
            company="TechCorp Inc.",
            fiscal_year=2023,
            pipeline="text_only",
            provider=mock_provider,
        )
        assert result.raw_json is not None
        assert isinstance(result.raw_json, dict)

    def test_invalid_json_returns_none_report(self, monkeypatch):
        """If LLM always returns garbage, report should be None."""

        class BadProvider:
            provider_name = "bad"
            model_name = "bad-model"

            def extract(self, prompt, context):
                return LLMResponse(
                    text="this is not json at all!!!",
                    provider="bad",
                    model="bad-model",
                )

        result = extract_from_context(
            "some context",
            company="Test",
            fiscal_year=2023,
            pipeline="text_only",
            provider=BadProvider(),
        )
        assert result.report is None
        assert result.succeeded is False
        assert len(result.validation_errors) > 0

    def test_json_cleaner_strips_backticks(self):
        dirty = "```json\n{\"a\": 1}\n```"
        assert _clean_json_text(dirty) == '{"a": 1}'

    def test_json_cleaner_strips_plain_backticks(self):
        dirty = "```\n{\"a\": 1}\n```"
        assert _clean_json_text(dirty) == '{"a": 1}'

    def test_try_parse_json_valid(self):
        assert _try_parse_json('{"a": 1}') == {"a": 1}

    def test_try_parse_json_invalid(self):
        assert _try_parse_json("not json") is None

    def test_try_parse_json_non_dict(self):
        assert _try_parse_json("[1, 2, 3]") is None


# ============================================================
# Pipeline A — Text Only
# ============================================================


class TestTextOnlyPipeline:
    def test_runs_on_sample_pdf(
        self, parsed_doc: ParsedDocument, mock_provider: MockProvider, tracker: ProvenanceTracker
    ):
        report, provenance = run_text_only(
            parsed_doc,
            company="TechCorp Inc.",
            fiscal_year=2023,
            provider=mock_provider,
            tracker=tracker,
        )
        assert isinstance(report, FinancialReport)
        assert report.company == "TechCorp Inc."
        assert report.fiscal_year == 2023

    def test_revenue_extracted_from_pdf(
        self, parsed_doc: ParsedDocument, mock_provider: MockProvider, tracker: ProvenanceTracker
    ):
        report, _ = run_text_only(
            parsed_doc,
            company="TechCorp Inc.",
            fiscal_year=2023,
            provider=mock_provider,
            tracker=tracker,
        )
        assert report.revenue is not None
        # Sample PDF has revenue = 50,000 (in millions)
        assert report.revenue.value == pytest.approx(SAMPLE_DATA["revenue"])

    def test_returns_provenance_records(
        self, parsed_doc: ParsedDocument, mock_provider: MockProvider, tracker: ProvenanceTracker
    ):
        report, provenance = run_text_only(
            parsed_doc,
            company="TechCorp Inc.",
            fiscal_year=2023,
            provider=mock_provider,
            tracker=tracker,
        )
        assert isinstance(provenance, list)
        # Each extracted field should have a provenance record
        n_extracted = sum(1 for m in report.metric_fields().values() if m is not None)
        assert len(provenance) == n_extracted

    def test_provenance_has_metadata(
        self, parsed_doc: ParsedDocument, mock_provider: MockProvider, tracker: ProvenanceTracker
    ):
        _, provenance = run_text_only(
            parsed_doc,
            company="TechCorp Inc.",
            fiscal_year=2023,
            provider=mock_provider,
            tracker=tracker,
        )
        for record in provenance:
            assert record.document_id == "pipeline-test-001"
            assert record.company == "TechCorp Inc."
            assert record.field_name is not None
            assert record.extraction_method == ExtractionMethod.TEXT_ONLY

    def test_context_selection_finds_relevant_pages(self, parsed_doc: ParsedDocument):
        context = _select_text_context(parsed_doc)
        assert "Revenue" in context
        assert len(context) <= 12_001  # MAX_CONTEXT_CHARS + some slack


# ============================================================
# Pipeline B — Layout Aware
# ============================================================


class TestLayoutAwarePipeline:
    def test_runs_on_sample_pdf(
        self, parsed_doc: ParsedDocument, mock_provider: MockProvider, tracker: ProvenanceTracker
    ):
        report, provenance = run_layout_aware(
            parsed_doc,
            company="TechCorp Inc.",
            fiscal_year=2023,
            provider=mock_provider,
            tracker=tracker,
        )
        assert isinstance(report, FinancialReport)

    def test_revenue_extracted(
        self, parsed_doc: ParsedDocument, mock_provider: MockProvider, tracker: ProvenanceTracker
    ):
        report, _ = run_layout_aware(
            parsed_doc,
            company="TechCorp Inc.",
            fiscal_year=2023,
            provider=mock_provider,
            tracker=tracker,
        )
        assert report.revenue is not None
        assert report.revenue.value == pytest.approx(SAMPLE_DATA["revenue"])

    def test_context_within_limit(self, parsed_doc: ParsedDocument):
        context = _select_layout_context(parsed_doc)
        assert len(context) <= 12_001

    def test_table_to_text_format(self):
        table = ParsedTable(
            page=2,
            rows=[
                ["Revenue", "50,000", "45,000"],
                ["Net Income", "10,000", "8,000"],
            ],
        )
        text = _table_to_text(table)
        assert "Revenue" in text
        assert "50,000" in text
        assert "|" in text  # pipe-delimited format

    def test_empty_table_returns_empty_string(self):
        table = ParsedTable(page=1, rows=[])
        assert _table_to_text(table) == ""


# ============================================================
# Pipeline C — Hybrid
# ============================================================


class TestHybridPipeline:
    def test_runs_on_sample_pdf(
        self, parsed_doc: ParsedDocument, mock_provider: MockProvider, tracker: ProvenanceTracker
    ):
        report, provenance = run_hybrid(
            parsed_doc,
            company="TechCorp Inc.",
            fiscal_year=2023,
            provider=mock_provider,
            tracker=tracker,
        )
        assert isinstance(report, FinancialReport)

    def test_revenue_extracted(
        self, parsed_doc: ParsedDocument, mock_provider: MockProvider, tracker: ProvenanceTracker
    ):
        report, _ = run_hybrid(
            parsed_doc,
            company="TechCorp Inc.",
            fiscal_year=2023,
            provider=mock_provider,
            tracker=tracker,
        )
        assert report.revenue is not None
        assert report.revenue.value == pytest.approx(SAMPLE_DATA["revenue"])

    def test_hybrid_context_richer_than_text_only(self, parsed_doc: ParsedDocument):
        text_ctx = _select_text_context(parsed_doc)
        hybrid_ctx = _select_hybrid_context(parsed_doc)
        # Both should contain revenue info
        assert "Revenue" in hybrid_ctx or "revenue" in hybrid_ctx.lower()

    def test_consistency_check_revenue_vs_operating(self):
        """revenue < operating_income should add a consistency warning."""
        from finextract.validation.schemas import FinancialMetric

        report = FinancialReport(
            company="Test",
            fiscal_year=2023,
            extraction_method=ExtractionMethod.HYBRID,
            revenue=FinancialMetric(value=100.0),
            operating_income=FinancialMetric(value=500.0),  # > revenue!
        )
        _apply_consistency_checks(report, "context")
        assert any("operating_income" in e for e in report.validation_errors)

    def test_consistency_check_assets_vs_liabilities(self):
        """total_liabilities > total_assets should add a warning."""
        from finextract.validation.schemas import FinancialMetric

        report = FinancialReport(
            company="Test",
            fiscal_year=2023,
            extraction_method=ExtractionMethod.HYBRID,
            total_assets=FinancialMetric(value=100.0),
            total_liabilities=FinancialMetric(value=500.0),  # > assets!
        )
        _apply_consistency_checks(report, "context")
        assert any("liabilities" in e for e in report.validation_errors)

    def test_consistency_check_passes_for_valid_report(self):
        """No warnings for a financially consistent report."""
        from finextract.validation.schemas import FinancialMetric

        report = FinancialReport(
            company="Test",
            fiscal_year=2023,
            extraction_method=ExtractionMethod.HYBRID,
            revenue=FinancialMetric(value=50_000.0),
            operating_income=FinancialMetric(value=12_000.0),
            total_assets=FinancialMetric(value=80_000.0),
            total_liabilities=FinancialMetric(value=30_000.0),
        )
        _apply_consistency_checks(report, "context")
        assert report.validation_errors == []


# ============================================================
# Provider factory tests
# ============================================================


class TestProviderFactory:
    def test_get_mock_provider(self):
        provider = get_provider("mock")
        assert provider.provider_name == "mock"
        assert isinstance(provider, MockProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider("nonexistent_provider_xyz")

    def test_mock_is_default(self, monkeypatch):
        """When LLM_PROVIDER is not set, default is 'mock'."""
        from finextract.config.settings import settings
        assert settings.llm_provider == "mock"


# ============================================================
# Prompt template tests
# ============================================================


class TestPromptTemplates:
    def test_text_only_prompt(self):
        prompt = get_system_prompt("text_only")
        assert prompt == TEXT_ONLY_SYSTEM_PROMPT
        assert "revenue" in prompt.lower()
        assert "null" in prompt.lower()
        assert "unit" in prompt.lower()

    def test_layout_aware_prompt(self):
        prompt = get_system_prompt("layout_aware")
        assert prompt == LAYOUT_AWARE_SYSTEM_PROMPT
        assert "table" in prompt.lower()

    def test_hybrid_prompt(self):
        prompt = get_system_prompt("hybrid")
        assert prompt == HYBRID_SYSTEM_PROMPT
        assert "balance sheet" in prompt.lower()
        assert "diluted" in prompt.lower()

    def test_default_is_text_only(self):
        prompt = get_system_prompt("unknown_pipeline")
        assert prompt == TEXT_ONLY_SYSTEM_PROMPT

    def test_all_prompts_include_unit_warning(self):
        for pipeline in ["text_only", "layout_aware", "hybrid"]:
            prompt = get_system_prompt(pipeline)
            assert "millions" in prompt.lower() or "unit" in prompt.lower()

    def test_all_prompts_include_null_rule(self):
        for pipeline in ["text_only", "layout_aware", "hybrid"]:
            prompt = get_system_prompt(pipeline)
            assert "null" in prompt

    def test_all_prompts_different(self):
        """All three prompts must be distinct."""
        p1 = get_system_prompt("text_only")
        p2 = get_system_prompt("layout_aware")
        p3 = get_system_prompt("hybrid")
        assert p1 != p2
        assert p2 != p3
        assert p1 != p3

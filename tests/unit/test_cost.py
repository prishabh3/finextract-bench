"""
Unit tests for finextract.evaluation.cost.

Covers:
- estimate_cost with pricing available
- estimate_cost with no pricing for model
- estimate_cost with None usage
- aggregate_cost with mixed None/float costs
- PricingRegistry loading and prefix matching
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from finextract.evaluation.cost import (
    CostEstimate,
    PricingRegistry,
    TokenUsage,
    aggregate_cost,
    estimate_cost,
)

# ============================================================
# Fixtures
# ============================================================


SAMPLE_PRICING_YAML = textwrap.dedent("""
    models:
      gpt-4o:
        input_cost_per_1m_tokens: 2.50
        output_cost_per_1m_tokens: 10.00
      mock-model-v1:
        input_cost_per_1m_tokens: 0.0
        output_cost_per_1m_tokens: 0.0
      claude-3-5-sonnet:
        input_cost_per_1m_tokens: 3.00
        output_cost_per_1m_tokens: 15.00
""")


@pytest.fixture()
def registry(tmp_path: Path) -> PricingRegistry:
    """PricingRegistry loaded from a temp file with sample pricing."""
    pricing_file = tmp_path / "model_pricing.yaml"
    pricing_file.write_text(SAMPLE_PRICING_YAML)
    return PricingRegistry(pricing_file=pricing_file)


@pytest.fixture()
def empty_registry(tmp_path: Path) -> PricingRegistry:
    """PricingRegistry with no models configured."""
    pricing_file = tmp_path / "model_pricing.yaml"
    pricing_file.write_text("models: {}")
    return PricingRegistry(pricing_file=pricing_file)


# ============================================================
# PricingRegistry
# ============================================================


class TestPricingRegistry:
    def test_exact_model_match(self, registry: PricingRegistry):
        pricing = registry.get_pricing("gpt-4o")
        assert pricing is not None
        assert pricing["input_cost_per_1m_tokens"] == pytest.approx(2.50)
        assert pricing["output_cost_per_1m_tokens"] == pytest.approx(10.00)

    def test_prefix_match(self, registry: PricingRegistry):
        """gpt-4o-2024-05-13 should match gpt-4o."""
        pricing = registry.get_pricing("gpt-4o-2024-05-13")
        assert pricing is not None

    def test_unknown_model_returns_none(self, registry: PricingRegistry):
        pricing = registry.get_pricing("unknown-model-xyz")
        assert pricing is None

    def test_missing_pricing_file(self, tmp_path: Path):
        """Missing file should produce empty registry without raising."""
        reg = PricingRegistry(pricing_file=tmp_path / "nonexistent.yaml")
        assert reg.get_pricing("gpt-4o") is None

    def test_mock_model_zero_cost(self, registry: PricingRegistry):
        pricing = registry.get_pricing("mock-model-v1")
        assert pricing is not None
        assert pricing["input_cost_per_1m_tokens"] == pytest.approx(0.0)


# ============================================================
# estimate_cost
# ============================================================


class TestEstimateCost:
    def test_known_model(self, registry: PricingRegistry):
        usage = TokenUsage(
            input_tokens=1_000_000,  # 1M tokens
            output_tokens=1_000_000,
            provider="openai",
            model="gpt-4o",
        )
        result = estimate_cost(usage, registry=registry)
        # 1M input * $2.50/M + 1M output * $10.00/M = $12.50
        assert result.cost_usd == pytest.approx(12.50)
        assert result.pricing_available is True
        assert result.input_tokens == 1_000_000

    def test_small_usage(self, registry: PricingRegistry):
        usage = TokenUsage(
            input_tokens=500,
            output_tokens=100,
            provider="openai",
            model="gpt-4o",
        )
        result = estimate_cost(usage, registry=registry)
        # 500/1M * 2.50 + 100/1M * 10.00 = 0.00125 + 0.001 = 0.00225
        expected = 500 / 1_000_000 * 2.50 + 100 / 1_000_000 * 10.00
        assert result.cost_usd == pytest.approx(expected)

    def test_mock_model_zero_cost(self, registry: PricingRegistry):
        usage = TokenUsage(
            input_tokens=10_000,
            output_tokens=500,
            provider="mock",
            model="mock-model-v1",
        )
        result = estimate_cost(usage, registry=registry)
        assert result.cost_usd == pytest.approx(0.0)
        assert result.pricing_available is True

    def test_unknown_model_no_cost(self, registry: PricingRegistry):
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=200,
            provider="some-provider",
            model="unknown-future-model",
        )
        result = estimate_cost(usage, registry=registry)
        assert result.cost_usd is None
        assert result.pricing_available is False
        assert result.input_tokens == 1000

    def test_none_usage_returns_none_cost(self, registry: PricingRegistry):
        result = estimate_cost(None, registry=registry)
        assert result.cost_usd is None
        assert result.pricing_available is False
        assert result.input_tokens is None

    def test_anthropic_model(self, registry: PricingRegistry):
        usage = TokenUsage(
            input_tokens=2_000_000,
            output_tokens=500_000,
            provider="anthropic",
            model="claude-3-5-sonnet",
        )
        result = estimate_cost(usage, registry=registry)
        # 2M * 3.00 + 0.5M * 15.00 = 6.00 + 7.50 = 13.50
        assert result.cost_usd == pytest.approx(13.50)


# ============================================================
# aggregate_cost
# ============================================================


class TestAggregateCost:
    def test_all_known(self):
        estimates = [
            CostEstimate(input_tokens=100, output_tokens=50, cost_usd=0.01,
                         model="m", provider="p", pricing_available=True),
            CostEstimate(input_tokens=200, output_tokens=100, cost_usd=0.02,
                         model="m", provider="p", pricing_available=True),
        ]
        assert aggregate_cost(estimates) == pytest.approx(0.03)

    def test_all_none(self):
        estimates = [
            CostEstimate(input_tokens=None, output_tokens=None, cost_usd=None,
                         model="m", provider="p", pricing_available=False),
        ]
        assert aggregate_cost(estimates) is None

    def test_mixed(self):
        estimates = [
            CostEstimate(input_tokens=100, output_tokens=50, cost_usd=0.05,
                         model="m", provider="p", pricing_available=True),
            CostEstimate(input_tokens=None, output_tokens=None, cost_usd=None,
                         model="m", provider="p", pricing_available=False),
        ]
        # Only non-None costs summed
        assert aggregate_cost(estimates) == pytest.approx(0.05)

    def test_empty(self):
        assert aggregate_cost([]) is None

    def test_zero_cost(self):
        estimates = [
            CostEstimate(input_tokens=0, output_tokens=0, cost_usd=0.0,
                         model="mock", provider="mock", pricing_available=True),
        ]
        assert aggregate_cost(estimates) == pytest.approx(0.0)

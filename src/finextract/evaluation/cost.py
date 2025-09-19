"""
FinExtract-Bench: Provider-independent cost calculation.

Token costs are loaded from config/model_pricing.yaml. If a provider does
not expose token usage, cost is explicitly marked as unavailable (None)
rather than silently estimated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ============================================================
# Cost data structures
# ============================================================


@dataclass(frozen=True)
class TokenUsage:
    """Raw token counts returned by a provider."""

    input_tokens: int
    output_tokens: int
    provider: str
    model: str


@dataclass(frozen=True)
class CostEstimate:
    """
    Estimated monetary cost for one LLM call.

    cost_usd is None if the provider did not expose token usage or
    if no pricing is configured for the model.
    """

    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    model: str
    provider: str
    pricing_available: bool


# ============================================================
# Pricing registry
# ============================================================


class PricingRegistry:
    """
    Loads and queries model pricing from config/model_pricing.yaml.

    Pricing format (per model entry):
        input_cost_per_1m_tokens: float   # USD per 1 million input tokens
        output_cost_per_1m_tokens: float  # USD per 1 million output tokens
    """

    def __init__(self, pricing_file: Path | None = None) -> None:
        self._pricing: dict[str, dict[str, float]] = {}
        if pricing_file is None:
            pricing_file = (
                Path(__file__).resolve().parents[4] / "config" / "model_pricing.yaml"
            )
        self._load(pricing_file)

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Pricing file not found: %s. Cost estimation disabled.", path)
            return
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        self._pricing = data.get("models", {})
        logger.debug("Loaded pricing for %d models.", len(self._pricing))

    def get_pricing(self, model: str) -> dict[str, float] | None:
        """
        Return pricing for a model, or None if not configured.

        Supports both exact matches and prefix matches
        (e.g. 'gpt-4o' matches 'gpt-4o-2024-05-13').
        """
        if model in self._pricing:
            return self._pricing[model]
        # Prefix match
        for key in self._pricing:
            if model.startswith(key) or key.startswith(model):
                return self._pricing[key]
        return None


# Module-level singleton
_registry: PricingRegistry | None = None


def get_registry() -> PricingRegistry:
    """Return the module-level PricingRegistry, loading it on first access."""
    global _registry
    if _registry is None:
        _registry = PricingRegistry()
    return _registry


# ============================================================
# Core cost calculation
# ============================================================


def estimate_cost(
    usage: TokenUsage | None,
    *,
    registry: PricingRegistry | None = None,
) -> CostEstimate:
    """
    Estimate the USD cost for an LLM call.

    Args:
        usage: Token usage from the provider. If None (provider didn't report
               usage), cost_usd is set to None and pricing_available=False.
        registry: Pricing registry to query. Defaults to the module singleton.

    Returns:
        CostEstimate with cost_usd=None if pricing unavailable.
    """
    if registry is None:
        registry = get_registry()

    if usage is None:
        return CostEstimate(
            input_tokens=None,
            output_tokens=None,
            cost_usd=None,
            model="unknown",
            provider="unknown",
            pricing_available=False,
        )

    pricing = registry.get_pricing(usage.model)

    if pricing is None:
        logger.debug(
            "No pricing configured for model %r. Cost marked as unavailable.", usage.model
        )
        return CostEstimate(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=None,
            model=usage.model,
            provider=usage.provider,
            pricing_available=False,
        )

    input_cost = usage.input_tokens / 1_000_000 * pricing["input_cost_per_1m_tokens"]
    output_cost = usage.output_tokens / 1_000_000 * pricing["output_cost_per_1m_tokens"]
    total_cost = input_cost + output_cost

    return CostEstimate(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=total_cost,
        model=usage.model,
        provider=usage.provider,
        pricing_available=True,
    )


def aggregate_cost(estimates: list[CostEstimate]) -> float | None:
    """
    Sum non-None cost estimates.

    Returns None if no estimates have pricing available, so callers can
    distinguish 'zero cost' from 'cost unknown'.
    """
    costs = [e.cost_usd for e in estimates if e.cost_usd is not None]
    if not costs:
        return None
    return sum(costs)

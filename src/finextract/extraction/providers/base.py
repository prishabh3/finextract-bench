"""
FinExtract-Bench: LLM Provider abstraction layer.

Design principles:
- Providers are interchangeable via a common Protocol.
- Every call returns an LLMResponse with text + token counts.
- Token counts are None if the provider doesn't expose them (not silently 0).
- Retry logic lives here, not in callers.
- The mock provider is fully deterministic and needs no API key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ============================================================
# LLM response container
# ============================================================


@dataclass(frozen=True)
class LLMResponse:
    """
    Result of one LLM extraction call.

    Attributes:
        text: Raw text response from the model.
        input_tokens: Number of input tokens consumed, or None if unavailable.
        output_tokens: Number of output tokens generated, or None if unavailable.
        provider: Provider name (e.g. 'openai', 'anthropic', 'mock').
        model: Model identifier (e.g. 'gpt-4o', 'mock-model-v1').
        latency_ms: Round-trip time in milliseconds for this call.
    """

    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


# ============================================================
# LLM Provider Protocol
# ============================================================


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol for all LLM provider implementations.

    Any class that implements extract() and exposes provider_name and
    model_name satisfies this protocol. No inheritance required.

    To add a new provider:
      1. Create a class in extraction/providers/<name>.py
      2. Implement extract(), provider_name, and model_name
      3. Register it in get_provider() in this file
    """

    @property
    def provider_name(self) -> str:
        """Short identifier for this provider, e.g. 'openai'."""
        ...

    @property
    def model_name(self) -> str:
        """Model identifier, e.g. 'gpt-4o'."""
        ...

    def extract(self, prompt: str, context: str) -> LLMResponse:
        """
        Run a structured extraction.

        Args:
            prompt: The instruction prompt (system/user instructions).
            context: The document text or table content to extract from.

        Returns:
            LLMResponse with the model's text output and token counts.

        Raises:
            ProviderError: On unrecoverable API or parsing error.
        """
        ...


# ============================================================
# Exceptions
# ============================================================


class ProviderError(RuntimeError):
    """Raised when an LLM provider encounters an unrecoverable error."""


class JSONParseError(ValueError):
    """Raised when the LLM response cannot be parsed as valid JSON."""


# ============================================================
# Provider registry / factory
# ============================================================


def get_provider(provider_name: str | None = None, model: str | None = None) -> LLMProvider:
    """
    Instantiate an LLM provider by name.

    Args:
        provider_name: One of 'mock', 'openai', 'anthropic', 'google'.
                       Defaults to the value in application settings.
        model: Model identifier override.

    Returns:
        An LLMProvider instance ready to call.

    Raises:
        ValueError: If provider_name is not recognized.
        ImportError: If the provider's library is not installed.
    """
    from finextract.config.settings import settings

    name = (provider_name or settings.llm_provider).lower()
    mdl = model or settings.llm_model

    if name == "mock":
        from finextract.extraction.providers.mock import MockProvider
        return MockProvider(model=mdl)

    if name == "openai":
        from finextract.extraction.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model=mdl)

    if name == "anthropic":
        from finextract.extraction.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=mdl)

    raise ValueError(
        f"Unknown LLM provider: {name!r}. "
        "Supported providers: 'mock', 'openai', 'anthropic'."
    )

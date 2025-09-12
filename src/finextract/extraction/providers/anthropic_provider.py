"""
FinExtract-Bench: Anthropic Claude LLM provider.

Uses the anthropic Python SDK (>= 0.28). Sends a messages API request
with a structured JSON extraction prompt.

Token usage is read from response.usage (input_tokens, output_tokens).
"""

from __future__ import annotations

import logging
import time

from finextract.config.settings import settings
from finextract.extraction.providers.base import LLMResponse, ProviderError

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Anthropic Claude provider (claude-3-5-sonnet, claude-3-5-haiku, etc.)."""

    def __init__(self, model: str = "claude-3-5-haiku-20241022") -> None:
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "anthropic package is required. Install with: pip install anthropic"
                ) from exc

            api_key = settings.anthropic_api_key
            if not api_key:
                raise ProviderError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Add it to your .env file or set the environment variable."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def extract(self, prompt: str, context: str) -> LLMResponse:
        """
        Send a messages API request to Claude and return the response.

        Args:
            prompt: System instruction for structured extraction.
            context: Document text to extract from.

        Returns:
            LLMResponse with text, token counts, and latency.

        Raises:
            ProviderError: On API error or timeout.
        """
        client = self._get_client()
        t_start = time.monotonic()
        last_exc: Exception | None = None

        for attempt in range(settings.llm_max_retries):
            try:
                response = client.messages.create(
                    model=self._model,
                    max_tokens=2048,
                    temperature=0.0,
                    system=prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Document text:\n\n{context}\n\n"
                                "Respond with only valid JSON."
                            ),
                        }
                    ],
                )
                elapsed_ms = (time.monotonic() - t_start) * 1000

                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text

                usage = response.usage

                return LLMResponse(
                    text=text,
                    provider="anthropic",
                    model=self._model,
                    input_tokens=usage.input_tokens if usage else None,
                    output_tokens=usage.output_tokens if usage else None,
                    latency_ms=elapsed_ms,
                )

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Anthropic attempt %d/%d failed: %s",
                    attempt + 1, settings.llm_max_retries, exc,
                )
                if attempt < settings.llm_max_retries - 1:
                    time.sleep(2 ** attempt)

        raise ProviderError(
            f"Anthropic extraction failed after {settings.llm_max_retries} attempts: {last_exc}"
        )

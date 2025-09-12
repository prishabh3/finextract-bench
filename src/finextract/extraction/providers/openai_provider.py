"""
FinExtract-Bench: OpenAI LLM provider.

Uses the openai Python SDK (>= 1.35). Sends a chat completion request
with a structured JSON instruction prompt and parses the response.

Token usage is read from response.usage and recorded for cost estimation.
"""

from __future__ import annotations

import logging
import time

from finextract.config.settings import settings
from finextract.extraction.providers.base import LLMResponse, ProviderError

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """OpenAI chat completion provider (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model
        self._client = None  # Lazy-initialized

    def _get_client(self):
        """Lazy-init the OpenAI client (avoids import at module load time)."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for the OpenAI provider. "
                    "Install with: pip install openai"
                ) from exc

            api_key = settings.openai_api_key
            if not api_key:
                raise ProviderError(
                    "OPENAI_API_KEY is not set. "
                    "Add it to your .env file or set the environment variable."
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def extract(self, prompt: str, context: str) -> LLMResponse:
        """
        Send a chat completion request and return the response.

        Args:
            prompt: System/user instruction prompt.
            context: Document text to include in the user message.

        Returns:
            LLMResponse with text, token counts, and latency.

        Raises:
            ProviderError: On API error or timeout.
        """
        client = self._get_client()

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Document text:\n\n{context}"},
        ]

        t_start = time.monotonic()
        last_exc: Exception | None = None

        for attempt in range(settings.llm_max_retries):
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    timeout=settings.llm_timeout_seconds,
                    temperature=0.0,  # Deterministic extraction
                )
                elapsed_ms = (time.monotonic() - t_start) * 1000

                text = response.choices[0].message.content or ""
                usage = response.usage

                return LLMResponse(
                    text=text,
                    provider="openai",
                    model=self._model,
                    input_tokens=usage.prompt_tokens if usage else None,
                    output_tokens=usage.completion_tokens if usage else None,
                    latency_ms=elapsed_ms,
                )

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "OpenAI attempt %d/%d failed: %s",
                    attempt + 1, settings.llm_max_retries, exc,
                )
                if attempt < settings.llm_max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        raise ProviderError(
            f"OpenAI extraction failed after {settings.llm_max_retries} attempts: {last_exc}"
        )

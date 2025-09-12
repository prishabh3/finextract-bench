"""
FinExtract-Bench: Mock LLM provider — deterministic, no API key required.

The mock provider parses the context text for financial keyword+number
patterns and returns structured JSON. This enables the full pipeline to be
tested end-to-end without any paid API.

Design:
- Completely deterministic: same context → same output every time.
- Clearly labeled as mock in all provenance records.
- Does NOT fabricate values — it only returns numbers it finds in the context.
- If a field's keyword is not found, it returns null for that field.
- Token counts are approximated from string length (for cost estimation tests).

The mock uses a keyword → field mapping to identify values:

  "Revenue"             → revenue
  "Net Income"          → net_income
  "Operating Income"    → operating_income
  "Total Assets"        → total_assets
  "Total Liabilities"   → total_liabilities
  "Cash and Cash"       → cash_and_equivalents
  "Earnings Per Share"  → eps
"""

from __future__ import annotations

import json
import logging
import re
import time

from finextract.extraction.providers.base import LLMResponse

logger = logging.getLogger(__name__)


# ── Keyword → field mapping (ordered from most specific to least) ──────
_FIELD_KEYWORDS: list[tuple[str, str]] = [
    # (search phrase, field_name)
    ("Cash and Cash Equivalents", "cash_and_equivalents"),
    ("Cash and Equivalents", "cash_and_equivalents"),
    ("Cash & Cash Equivalents", "cash_and_equivalents"),
    ("Earnings Per Share (Diluted)", "eps"),
    ("Earnings Per Share", "eps"),
    ("EPS (Diluted)", "eps"),
    ("Diluted EPS", "eps"),
    ("Operating Income", "operating_income"),
    ("Income from Operations", "operating_income"),
    ("Total Assets", "total_assets"),
    ("Total Liabilities", "total_liabilities"),
    ("Net Income", "net_income"),
    ("Net Earnings", "net_income"),
    ("Net Profit", "net_income"),
    ("Total Revenue", "revenue"),
    ("Net Revenue", "revenue"),
    ("Revenue", "revenue"),
    ("Net Sales", "revenue"),
    ("Total Net Sales", "revenue"),
]

# Pattern to match a financial number immediately following a keyword
# Handles: 50,000  or  50,000.00  or  (50,000)  or  -50,000
_NUMBER_PATTERN = re.compile(
    r"[\t ]*:?[\t ]*\$?([\-\(]?[\d,]+(?:\.\d+)?\)?)",
    re.IGNORECASE,
)


class MockProvider:
    """
    Deterministic mock LLM provider for testing.

    Scans context text for financial label+value patterns and returns
    them as valid JSON. No API key or network call is made.
    """

    def __init__(self, model: str = "mock-model-v1") -> None:
        self._model = model

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    def extract(self, prompt: str, context: str) -> LLMResponse:
        """
        Scan context for financial values and return structured JSON.

        Args:
            prompt: Ignored for the mock (structure is implicit).
            context: Document text to scan for financial values.

        Returns:
            LLMResponse with JSON text containing extracted values or nulls.
        """
        t_start = time.monotonic()

        extracted = _scan_context(context)
        response_text = json.dumps(extracted, indent=2)

        # Approximate token counts for cost estimation tests
        input_tokens = _approx_tokens(prompt + context)
        output_tokens = _approx_tokens(response_text)

        elapsed_ms = (time.monotonic() - t_start) * 1000

        logger.debug(
            "Mock provider extracted %d/%d fields from context (%d chars)",
            sum(1 for v in extracted.values() if v is not None),
            len(extracted),
            len(context),
        )

        return LLMResponse(
            text=response_text,
            provider="mock",
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )


def _scan_context(context: str) -> dict[str, dict | None]:
    """
    Scan context text for financial keyword + value pairs.

    Handles two common layouts:
      1. Same-line: "Revenue: 50,000"
      2. Next-line:  "Revenue\n50,000"  (common in PyMuPDF text-block output)

    Returns a dict of {field_name: {value, unit, source_text} | None}.
    """
    result: dict[str, dict | None] = {
        "revenue": None,
        "net_income": None,
        "operating_income": None,
        "total_assets": None,
        "total_liabilities": None,
        "cash_and_equivalents": None,
        "eps": None,
    }

    already_found: set[str] = set()

    for keyword, field_name in _FIELD_KEYWORDS:
        if field_name in already_found:
            continue

        # Find keyword in context (case-insensitive)
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        match = pattern.search(context)
        if not match:
            continue

        # Search region: from end of keyword match, up to 300 chars
        after_keyword = context[match.end() : match.end() + 300]

        # Try: number on the same line (after optional whitespace/colon)
        same_line_end = after_keyword.find("\n")
        same_line = after_keyword[:same_line_end] if same_line_end != -1 else after_keyword
        num_match = _NUMBER_PATTERN.match(same_line)

        if not num_match:
            # Try: number on the NEXT line ("Revenue\n50,000")
            next_line_start = same_line_end + 1 if same_line_end != -1 else 0
            next_line_region = after_keyword[next_line_start:]
            next_line_end = next_line_region.find("\n")
            next_line = next_line_region[:next_line_end] if next_line_end != -1 else next_line_region
            num_match = _NUMBER_PATTERN.match(next_line)

            if not num_match:
                # Try: number on the line after next (some PDFs insert blank lines)
                after_next = next_line_region[next_line_end + 1:] if next_line_end != -1 else ""
                after_next_end = after_next.find("\n")
                after_next_line = after_next[:after_next_end] if after_next_end != -1 else after_next
                num_match = _NUMBER_PATTERN.match(after_next_line)

        if not num_match:
            continue

        raw_value = num_match.group(1).strip()
        source_start = max(0, match.start() - 20)
        source_end = min(len(context), match.end() + 80)
        source_text = context[source_start:source_end].strip()

        result[field_name] = {
            "value": raw_value,
            "unit": "million USD",  # assumption for the mock
            "currency": "USD",
            "source_text": source_text,
            "confidence": 0.70,  # mock confidence — clearly not real
        }
        already_found.add(field_name)

    return result


def _approx_tokens(text: str) -> int:
    """
    Approximate token count from character length.

    Rough heuristic: ~4 characters per token (GPT-style tokenization).
    Used only for cost estimation in tests — not for billing.
    """
    return max(1, len(text) // 4)

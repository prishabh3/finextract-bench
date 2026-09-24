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

  "Revenue" / "Net Sales"          → revenue
  "Net Income" / "Net Earnings"    → net_income
  "Operating Income"               → operating_income
  "Total Assets"                   → total_assets
  "Total Liabilities"              → total_liabilities
  "Cash and Cash"                  → cash_and_equivalents
  "Earnings Per Share"             → eps
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
    # Cash
    ("Cash, Cash Equivalents and Restricted Cash", "cash_and_equivalents"),
    ("Cash and Cash Equivalents", "cash_and_equivalents"),
    ("Cash and Equivalents", "cash_and_equivalents"),
    ("Cash & Cash Equivalents", "cash_and_equivalents"),
    ("Cash, cash equivalents", "cash_and_equivalents"),
    ("Cash and short-term investments", "cash_and_equivalents"),
    # EPS — order matters: most specific first
    ("Diluted earnings per share", "eps"),
    ("Earnings Per Share (Diluted)", "eps"),
    ("EPS (Diluted)", "eps"),
    ("Diluted EPS", "eps"),
    ("Basic and diluted net income per share", "eps"),
    ("Net income per share", "eps"),
    ("Earnings Per Share", "eps"),
    ("Earnings per share", "eps"),
    # Operating Income
    ("Operating Income", "operating_income"),
    ("Income from Operations", "operating_income"),
    ("Operating income/(loss)", "operating_income"),
    ("Income from operations", "operating_income"),
    # Assets & Liabilities (must come before simpler patterns)
    ("Total Assets", "total_assets"),
    ("Total assets", "total_assets"),
    ("TOTAL ASSETS", "total_assets"),
    ("Total Liabilities", "total_liabilities"),
    ("Total liabilities", "total_liabilities"),
    ("TOTAL LIABILITIES", "total_liabilities"),
    # Net Income
    ("Net Income", "net_income"),
    ("Net income", "net_income"),
    ("Net Earnings", "net_income"),
    ("Net earnings", "net_income"),
    ("Net Profit", "net_income"),
    ("NET INCOME", "net_income"),
    ("Net income (loss)", "net_income"),
    # Revenue (last — broadest match)
    ("Total Net Revenue", "revenue"),
    ("Total net revenue", "revenue"),
    ("Total Net Sales", "revenue"),
    ("Total net sales", "revenue"),
    ("Net Revenue", "revenue"),
    ("Net revenue", "revenue"),
    ("Net Sales", "revenue"),
    ("Net sales", "revenue"),
    ("Total Revenue", "revenue"),
    ("Total revenue", "revenue"),
    ("TOTAL REVENUE", "revenue"),
    ("Revenue", "revenue"),
    ("Revenues", "revenue"),
    ("REVENUE", "revenue"),
    ("Sales", "revenue"),
]

# Robust number pattern: matches financial numbers in various formats
# Handles: 50,000  |  50,000.00  |  (50,000)  |  -50,000  |  $50,000
# Also matches: 394,328  |  1.29  |  (2,345.67)  |  $ 12,345
_NUMBER_RE = re.compile(
    r"""
    \$?\s*                          # optional dollar sign + space
    (\(?\s*-?\s*                    # optional opening paren or minus
     \d[\d,]*                       # digits with optional commas
     (?:\.\d+)?                     # optional decimal
     \s*\)?)                        # optional closing paren
    """,
    re.VERBOSE,
)

# Pattern to find a number anywhere in a chunk of text (for flexible matching)
_FIND_NUMBER_RE = re.compile(
    r"""
    \$?\s*                          # optional dollar sign
    (\(?\s*-?\s*                    # optional paren/minus
     \d[\d,]*                       # digits with commas
     (?:\.\d+)?                     # optional decimal
     \s*\)?)                        # optional closing paren
    """,
    re.VERBOSE,
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

    Handles multiple common layouts found in real financial PDFs:
      1. Same-line colon:   "Revenue: 50,000"
      2. Same-line tab:     "Revenue    50,000    45,000"
      3. Next-line:         "Revenue\n50,000"
      4. Spaced columns:   "Net sales                    394,328"
      5. Markdown tables:  "| Revenue | 50,000 | 45,000 |"

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

        # Search region: from end of keyword match, up to 500 chars
        search_end = min(len(context), match.end() + 500)
        after_keyword = context[match.end():search_end]

        # Split into individual lines and search each one for a number
        # Apple 10-K format example:
        #   Total net sales\n \n416,161 \n \n391,035
        # So we need to skip blank lines and lines that only contain "$" or whitespace
        lines = after_keyword.split("\n")
        num_match = None

        for line in lines[:8]:  # Check up to 8 lines ahead
            stripped = line.strip()

            # Skip empty lines, lines that are just "$" or "$ ", or section headers
            if not stripped or stripped in ("$", "$ ") or stripped == " ":
                continue

            # Remove leading "$ " that Apple uses before numbers
            cleaned_line = re.sub(r"^\s*\$\s*", "", stripped)

            num_match = _FIND_NUMBER_RE.search(cleaned_line)
            if num_match:
                break

            # Also try the original line (for formats like "$307,003")
            num_match = _FIND_NUMBER_RE.search(stripped)
            if num_match:
                break

        if not num_match:
            continue

        raw_value = num_match.group(1).strip()

        # Skip values that are clearly not financial (e.g., page numbers, years)
        cleaned = raw_value.replace(",", "").replace("(", "").replace(")", "").replace("-", "").strip()
        if not cleaned:
            continue

        try:
            numeric_val = float(cleaned)
        except ValueError:
            continue

        # Skip year-like values (2000-2100) or tiny page numbers
        if 2000 <= numeric_val <= 2100:
            continue

        # Build source text snippet for provenance
        source_start = max(0, match.start() - 20)
        source_end = min(len(context), match.end() + 120)
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

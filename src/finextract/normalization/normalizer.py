"""
FinExtract-Bench: Deterministic financial value normalization.

This module converts raw string representations of financial figures into
normalized (value, unit) pairs. All normalization is deterministic and
well-tested — no LLM or heuristic guessing is used here.

Design notes:
- Unit tracking is explicit and conservative. If a unit is ambiguous,
  the function raises NormalizationError rather than silently guessing.
- "millions" vs "thousands" vs bare values are treated as distinct units.
  A caller must decide how to reconcile units across fields.
- Parentheses are treated as negative values (standard accounting notation).
- Currency symbols are stripped and returned as a separate currency code.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================


class NormalizationError(ValueError):
    """Raised when a value string cannot be reliably normalized."""


# ============================================================
# Data class for normalized output
# ============================================================


@dataclass(frozen=True)
class NormalizedValue:
    """
    Result of normalizing a raw financial string.

    Attributes:
        value: Floating-point numeric value after all transformations.
        unit: Human-readable unit string, e.g. 'million USD', 'USD per share'.
        currency: ISO 4217 currency code extracted from the string, or None.
        original: The exact input string, preserved for provenance.
        is_negative: Whether the value is negative (parentheses or minus sign).
        multiplier: The scale factor applied (1, 1e3, 1e6, 1e9).
    """

    value: float
    unit: str
    currency: str | None
    original: str
    is_negative: bool
    multiplier: float


# ============================================================
# Constants
# ============================================================

# Maps currency symbols to ISO 4217 codes
_CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₩": "KRW",
    "₣": "CHF",
    "CA$": "CAD",
    "A$": "AUD",
    "HK$": "HKD",
    "US$": "USD",
}

# Maps written-out scale words to multipliers
_SCALE_WORDS: dict[str, float] = {
    "trillion": 1e12,
    "trillions": 1e12,
    "billion": 1e9,
    "billions": 1e9,
    "bn": 1e9,
    "b": 1e9,  # Only matched as a word, not a letter
    "million": 1e6,
    "millions": 1e6,
    "mn": 1e6,
    "mm": 1e6,  # Accounting shorthand
    "m": 1e6,  # Only matched as a word suffix
    "thousand": 1e3,
    "thousands": 1e3,
    "k": 1e3,
}

# Regex components
_CURRENCY_SYM_PATTERN = r"(?:US\$|CA\$|A\$|HK\$|\$|€|£|¥|₹|₩|₣)"
_NUMBER_CORE = r"[\d,\.]+"
_SCALE_PATTERN = (
    r"(?:trillion|trillions|billion|billions|bn|million|millions|mn|mm|thousand|thousands)"
)


def _extract_currency_symbol(raw: str) -> tuple[str, str | None]:
    """
    Strip a leading currency symbol from the string.

    Returns:
        (remaining_string, iso_code_or_None)
    """
    raw = raw.strip()
    # Try multi-char symbols first (longest match)
    for symbol, code in sorted(_CURRENCY_SYMBOLS.items(), key=lambda x: -len(x[0])):
        if raw.startswith(symbol):
            return raw[len(symbol) :].strip(), code
    return raw, None


def _detect_negativity(raw: str) -> tuple[str, bool]:
    """
    Detect and remove accounting-negative notation.

    Handles:
      - Leading minus: -12,500
      - Parentheses: (12,500)

    Returns:
        (cleaned_string_without_sign, is_negative)
    """
    raw = raw.strip()
    is_negative = False

    # Parentheses notation: (value)
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()
        is_negative = True
    # Standard minus
    elif raw.startswith("-"):
        raw = raw[1:].strip()
        is_negative = True

    return raw, is_negative


def _extract_scale(raw: str) -> tuple[str, float]:
    """
    Extract a trailing scale word (million, billion, etc.) from the string.

    This is conservative — it only matches whole words at the end of the string.

    Returns:
        (string_without_scale, multiplier)

    Raises:
        NormalizationError: if a scale-like word is found mid-string in an
            ambiguous position.
    """
    # Match a scale word at the end of the string (case-insensitive)
    pattern = re.compile(
        rf"(?i)\b({_SCALE_PATTERN})\s*$",
        re.IGNORECASE,
    )
    match = pattern.search(raw)
    if match:
        word = match.group(1).lower()
        multiplier = _SCALE_WORDS[word]
        raw = raw[: match.start()].strip()
        return raw, multiplier

    # Also check trailing single letters: 416.161B, $2.5M
    trailing_letter = re.compile(r"([kmb])$", re.IGNORECASE)
    m2 = trailing_letter.search(raw)
    if m2:
        letter = m2.group(1).lower()
        # Explicit map for single letters to avoid false positives
        letter_map = {"k": 1e3, "m": 1e6, "b": 1e9}
        multiplier = letter_map[letter]
        raw = raw[: m2.start()].strip()
        return raw, multiplier

    return raw, 1.0


def _parse_number(raw: str) -> float:
    """
    Parse a plain numeric string with optional commas.

    Examples: '416,161' → 416161.0, '416.161' → 416.161, '416,161.5' → 416161.5

    Raises:
        NormalizationError: on parse failure.
    """
    # Remove commas (thousand separators)
    cleaned = raw.replace(",", "").strip()

    if not cleaned:
        raise NormalizationError(f"Empty numeric string after stripping: {raw!r}")

    try:
        return float(cleaned)
    except ValueError as exc:
        raise NormalizationError(
            f"Cannot parse numeric value from {raw!r} (cleaned: {cleaned!r})"
        ) from exc


def normalize_financial_value(
    raw: str,
    *,
    default_currency: str | None = "USD",
    base_unit_label: str = "USD",
) -> NormalizedValue:
    """
    Convert a raw financial string into a normalized (value, unit, currency) tuple.

    This is the primary normalization entry point. It handles all common
    financial string representations:

    Examples::

        >>> normalize_financial_value("$416,161")
        NormalizedValue(value=416161.0, unit='USD', currency='USD', ...)

        >>> normalize_financial_value("416,161 million")
        NormalizedValue(value=416161000000.0, unit='million USD', ...)

        >>> normalize_financial_value("416.161 billion")
        NormalizedValue(value=416161000000.0, unit='billion USD', ...)

        >>> normalize_financial_value("(12,500)")
        NormalizedValue(value=-12500.0, is_negative=True, ...)

        >>> normalize_financial_value("-12,500")
        NormalizedValue(value=-12500.0, is_negative=True, ...)

    Args:
        raw: The raw string from the document.
        default_currency: Currency code to use if none found in the string.
        base_unit_label: Base unit name (before multiplier prefix) for unit string.

    Returns:
        NormalizedValue with fully resolved value, unit, and currency.

    Raises:
        NormalizationError: On ambiguous or unparseable input.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise NormalizationError(f"Input must be a non-empty string, got: {raw!r}")

    original = raw
    raw = raw.strip()

    # Step 1: strip currency symbol (before negativity so '$' in '($1,000)' is handled later)
    # We need to handle the case where currency is inside parens: ($1,000)
    # Strategy: strip outer parens first if present, then currency, then scale.

    # Step 1a: detect and strip negativity FIRST so we can then process contents
    raw, is_negative = _detect_negativity(raw)

    # Step 1b: strip currency symbol
    raw, currency = _extract_currency_symbol(raw)
    if currency is None:
        currency = default_currency

    # Step 2: extract scale word / letter (must come before _parse_number)
    raw, multiplier = _extract_scale(raw)

    # Step 3: if there are still unmatched parens (e.g. after stripping scale from
    # '(12,500) million' the parens were not caught because the string didn't end with ')'),
    # try stripping negativity again now that the scale word is gone.
    if not is_negative:
        raw, is_negative = _detect_negativity(raw)

    # Step 4: parse the remaining numeric part
    numeric_value = _parse_number(raw)

    # Step 5: apply multiplier
    value = numeric_value * multiplier
    if is_negative:
        value = -value

    # Step 6: construct unit string
    # Examples: "million USD", "billion USD", "USD"
    if multiplier == 1e12:
        scale_label = "trillion "
    elif multiplier == 1e9:
        scale_label = "billion "
    elif multiplier == 1e6:
        scale_label = "million "
    elif multiplier == 1e3:
        scale_label = "thousand "
    else:
        scale_label = ""

    unit = f"{scale_label}{base_unit_label}".strip()

    return NormalizedValue(
        value=value,
        unit=unit,
        currency=currency,
        original=original,
        is_negative=is_negative,
        multiplier=multiplier,
    )


def normalize_currency_code(raw: str) -> str:
    """
    Normalize a currency string to an ISO 4217 code.

    Accepts symbols ($, €, £), written names (USD, US Dollar),
    and common variants.

    Raises:
        NormalizationError: if the string cannot be resolved to a known currency.
    """
    raw = raw.strip()

    # Already a valid 3-letter code?
    if re.match(r"^[A-Z]{3}$", raw.upper()):
        return raw.upper()

    # Symbol lookup
    if raw in _CURRENCY_SYMBOLS:
        return _CURRENCY_SYMBOLS[raw]

    # Written-name lookup
    written_map = {
        "us dollar": "USD",
        "us dollars": "USD",
        "dollar": "USD",
        "dollars": "USD",
        "euro": "EUR",
        "euros": "EUR",
        "pound": "GBP",
        "pounds": "GBP",
        "yen": "JPY",
        "rupee": "INR",
        "rupees": "INR",
    }
    lower = raw.lower()
    if lower in written_map:
        return written_map[lower]

    raise NormalizationError(
        f"Cannot resolve currency {raw!r} to a known ISO 4217 code."
    )


def normalize_unit(raw: str) -> str:
    """
    Normalize a raw unit string to a canonical form.

    Examples:
        'millions' → 'million'
        'bn' → 'billion'
        'USD millions' → 'million USD'

    This is a best-effort normalization. For strict evaluation, rely on
    the unit embedded in NormalizedValue returned by normalize_financial_value().
    """
    raw = raw.strip().lower()

    canonical: dict[str, str] = {
        "millions": "million",
        "mn": "million",
        "mm": "million",
        "billions": "billion",
        "bn": "billion",
        "thousands": "thousand",
        "k": "thousand",
    }
    return canonical.get(raw, raw)


def is_valid_financial_value(value: float) -> bool:
    """Return True if value is a finite number (not NaN or Inf)."""
    import math

    return not (math.isnan(value) or math.isinf(value))

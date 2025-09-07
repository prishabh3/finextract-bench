"""
Unit tests for finextract.normalization.normalizer.

Covers:
- Currency symbol extraction
- Scale word parsing (million, billion, thousand)
- Trailing letter scale (M, B, K)
- Parentheses as negative values
- Leading minus as negative
- Comma removal
- Combined representations
- NormalizationError on invalid input
- is_valid_financial_value
"""

from __future__ import annotations

import pytest

from finextract.normalization.normalizer import (
    NormalizationError,
    is_valid_financial_value,
    normalize_currency_code,
    normalize_financial_value,
)

# ============================================================
# normalize_financial_value — basic values
# ============================================================


class TestNormalizeBasicValues:
    def test_plain_integer(self):
        result = normalize_financial_value("416161")
        assert result.value == pytest.approx(416161.0)
        assert result.multiplier == 1.0
        assert result.is_negative is False

    def test_comma_separated(self):
        result = normalize_financial_value("416,161")
        assert result.value == pytest.approx(416161.0)

    def test_decimal(self):
        result = normalize_financial_value("416.161")
        assert result.value == pytest.approx(416.161)

    def test_comma_and_decimal(self):
        result = normalize_financial_value("416,161.50")
        assert result.value == pytest.approx(416161.50)

    def test_zero(self):
        result = normalize_financial_value("0")
        assert result.value == pytest.approx(0.0)

    def test_large_number(self):
        result = normalize_financial_value("1,234,567,890")
        assert result.value == pytest.approx(1_234_567_890.0)


# ============================================================
# normalize_financial_value — currency symbols
# ============================================================


class TestCurrencyExtraction:
    def test_dollar_sign(self):
        result = normalize_financial_value("$416,161")
        assert result.currency == "USD"
        assert result.value == pytest.approx(416161.0)

    def test_euro_sign(self):
        result = normalize_financial_value("€500")
        assert result.currency == "EUR"
        assert result.value == pytest.approx(500.0)

    def test_us_dollar_prefix(self):
        result = normalize_financial_value("US$100")
        assert result.currency == "USD"
        assert result.value == pytest.approx(100.0)

    def test_no_currency_uses_default(self):
        result = normalize_financial_value("1000", default_currency="EUR")
        assert result.currency == "EUR"

    def test_no_currency_no_default(self):
        result = normalize_financial_value("1000", default_currency=None)
        assert result.currency is None


# ============================================================
# normalize_financial_value — scale words
# ============================================================


class TestScaleWords:
    def test_million(self):
        result = normalize_financial_value("416,161 million")
        assert result.value == pytest.approx(416_161 * 1e6)
        assert result.multiplier == 1e6
        assert "million" in result.unit

    def test_billion(self):
        result = normalize_financial_value("416.161 billion")
        assert result.value == pytest.approx(416.161 * 1e9)
        assert result.multiplier == 1e9

    def test_thousand(self):
        result = normalize_financial_value("500 thousand")
        assert result.value == pytest.approx(500_000.0)
        assert result.multiplier == 1e3

    def test_trillion(self):
        result = normalize_financial_value("1.5 trillion")
        assert result.value == pytest.approx(1.5e12)
        assert result.multiplier == 1e12

    def test_millions_plural(self):
        result = normalize_financial_value("200 millions")
        assert result.value == pytest.approx(200e6)

    def test_bn_abbreviation(self):
        result = normalize_financial_value("2.5 bn")
        assert result.value == pytest.approx(2.5e9)

    def test_mm_abbreviation(self):
        result = normalize_financial_value("500 mm")
        assert result.value == pytest.approx(500e6)

    def test_dollar_and_million(self):
        result = normalize_financial_value("$416.161 billion")
        assert result.value == pytest.approx(416.161e9)
        assert result.currency == "USD"

    def test_unit_string_format(self):
        """Unit string should be 'million USD' not just 'million'."""
        result = normalize_financial_value("$500 million")
        assert result.unit == "million USD"

    def test_bare_value_unit(self):
        result = normalize_financial_value("$1000")
        assert result.unit == "USD"


# ============================================================
# normalize_financial_value — trailing letter scales
# ============================================================


class TestTrailingLetterScales:
    def test_trailing_M(self):
        result = normalize_financial_value("500M")
        assert result.value == pytest.approx(500e6)

    def test_trailing_B(self):
        result = normalize_financial_value("2.5B")
        assert result.value == pytest.approx(2.5e9)

    def test_trailing_K(self):
        result = normalize_financial_value("750K")
        assert result.value == pytest.approx(750_000.0)

    def test_trailing_lowercase_m(self):
        result = normalize_financial_value("500m")
        assert result.value == pytest.approx(500e6)


# ============================================================
# normalize_financial_value — negative values
# ============================================================


class TestNegativeValues:
    def test_parentheses_negative(self):
        result = normalize_financial_value("(12,500)")
        assert result.value == pytest.approx(-12_500.0)
        assert result.is_negative is True

    def test_leading_minus(self):
        result = normalize_financial_value("-12,500")
        assert result.value == pytest.approx(-12_500.0)
        assert result.is_negative is True

    def test_parentheses_with_millions(self):
        result = normalize_financial_value("(12,500) million")
        assert result.value == pytest.approx(-12_500e6)
        assert result.is_negative is True

    def test_parentheses_with_dollar(self):
        result = normalize_financial_value("($1,000)")
        assert result.value == pytest.approx(-1000.0)
        assert result.is_negative is True
        assert result.currency == "USD"

    def test_zero_negative_is_zero(self):
        result = normalize_financial_value("(0)")
        assert result.value == pytest.approx(0.0)
        assert result.is_negative is True


# ============================================================
# normalize_financial_value — original preserved
# ============================================================


class TestOriginalPreservation:
    def test_original_stored(self):
        raw = "$416,161 million"
        result = normalize_financial_value(raw)
        assert result.original == raw


# ============================================================
# normalize_financial_value — error cases
# ============================================================


class TestNormalizationErrors:
    def test_empty_string_raises(self):
        with pytest.raises(NormalizationError):
            normalize_financial_value("")

    def test_whitespace_only_raises(self):
        with pytest.raises(NormalizationError):
            normalize_financial_value("   ")

    def test_non_string_raises(self):
        with pytest.raises(NormalizationError):
            normalize_financial_value(None)  # type: ignore

    def test_letters_only_raises(self):
        with pytest.raises(NormalizationError):
            normalize_financial_value("not a number")

    def test_just_currency_symbol_raises(self):
        with pytest.raises(NormalizationError):
            normalize_financial_value("$")


# ============================================================
# normalize_currency_code
# ============================================================


class TestNormalizeCurrencyCode:
    def test_usd_code(self):
        assert normalize_currency_code("USD") == "USD"

    def test_dollar_symbol(self):
        assert normalize_currency_code("$") == "USD"

    def test_euro_symbol(self):
        assert normalize_currency_code("€") == "EUR"

    def test_written_dollar(self):
        assert normalize_currency_code("dollar") == "USD"
        assert normalize_currency_code("US Dollar") == "USD"

    def test_lowercase_code(self):
        assert normalize_currency_code("usd") == "USD"

    def test_invalid_raises(self):
        with pytest.raises(NormalizationError):
            normalize_currency_code("INVALID_CURRENCY")


# ============================================================
# is_valid_financial_value
# ============================================================


class TestIsValidFinancialValue:
    def test_normal_value(self):
        assert is_valid_financial_value(100.0) is True

    def test_zero(self):
        assert is_valid_financial_value(0.0) is True

    def test_negative(self):
        assert is_valid_financial_value(-500.0) is True

    def test_nan(self):
        assert is_valid_financial_value(float("nan")) is False

    def test_inf(self):
        assert is_valid_financial_value(float("inf")) is False

    def test_neg_inf(self):
        assert is_valid_financial_value(float("-inf")) is False

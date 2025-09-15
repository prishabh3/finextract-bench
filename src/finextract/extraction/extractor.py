"""
FinExtract-Bench: Core extraction engine.

This module bridges the LLM response → Pydantic FinancialReport conversion.

Responsibilities:
  1. Send context to LLM provider and handle JSON parse errors.
  2. Map the raw LLM JSON → FinancialMetric instances.
  3. Normalize each value via the normalization module.
  4. Validate the assembled FinancialReport via Pydantic.
  5. Log all validation errors without silently swallowing them.
  6. Return both the report and raw LLM output (for debugging).

What this module does NOT do:
  - Select which text/tables to send as context (that's in the pipelines).
  - Record provenance (that's done by the pipeline after extraction).
  - Persist anything to the database.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime

from pydantic import ValidationError

from finextract.extraction.prompts import CORRECTION_PROMPT_TEMPLATE, get_system_prompt
from finextract.extraction.providers.base import LLMProvider, LLMResponse
from finextract.normalization.normalizer import NormalizationError, normalize_financial_value
from finextract.validation.schemas import (
    ExtractionMethod,
    FinancialMetric,
    FinancialReport,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


# ============================================================
# Extraction result container
# ============================================================


class ExtractionResult:
    """
    Container for one complete extraction attempt.

    Attributes:
        report: The Pydantic-validated FinancialReport (or None on total failure).
        llm_response: The raw LLMResponse from the provider.
        raw_json: The LLM's JSON output as a Python dict (or None on parse failure).
        validation_errors: List of validation error messages.
        normalization_errors: List of normalization error messages.
        total_time_ms: Total extraction time in milliseconds.
    """

    def __init__(
        self,
        report: FinancialReport | None,
        llm_response: LLMResponse,
        raw_json: dict | None = None,
        validation_errors: list[str] | None = None,
        normalization_errors: list[str] | None = None,
        total_time_ms: float = 0.0,
    ) -> None:
        self.report = report
        self.llm_response = llm_response
        self.raw_json = raw_json
        self.validation_errors = validation_errors or []
        self.normalization_errors = normalization_errors or []
        self.total_time_ms = total_time_ms

    @property
    def succeeded(self) -> bool:
        """True if extraction produced a valid FinancialReport."""
        return self.report is not None

    @property
    def had_validation_errors(self) -> bool:
        return bool(self.validation_errors)

    @property
    def had_normalization_errors(self) -> bool:
        return bool(self.normalization_errors)


# ============================================================
# Core extraction function
# ============================================================


def extract_from_context(
    context: str,
    *,
    company: str,
    fiscal_year: int,
    pipeline: str,
    provider: LLMProvider,
    document_id: str | None = None,
) -> ExtractionResult:
    """
    Run LLM extraction on a context string and return a validated FinancialReport.

    Args:
        context: The text or table content to extract from.
        company: Company name for the FinancialReport identity.
        fiscal_year: Fiscal year for the FinancialReport identity.
        pipeline: Pipeline identifier ('text_only', 'layout_aware', 'hybrid').
        provider: LLM provider instance to use.
        document_id: Optional document UUID for the report record.

    Returns:
        ExtractionResult with the report and all intermediate artifacts.
    """
    t_start = time.monotonic()
    prompt = get_system_prompt(pipeline)

    # ── Step 1: Call LLM ──────────────────────────────────────────────
    llm_response = provider.extract(prompt, context)
    raw_text = llm_response.text

    # ── Step 2: Parse JSON ────────────────────────────────────────────
    raw_json, json_errors = _parse_json_with_retry(raw_text, prompt, provider)

    if raw_json is None:
        # Total JSON failure — cannot proceed
        total_ms = (time.monotonic() - t_start) * 1000
        logger.error(
            "JSON parse failed for %s FY%s [%s]: %s",
            company, fiscal_year, pipeline, json_errors,
        )
        return ExtractionResult(
            report=None,
            llm_response=llm_response,
            validation_errors=[f"JSON parse error: {e}" for e in json_errors],
            total_time_ms=total_ms,
        )

    # ── Step 3: Map JSON → FinancialMetric instances ──────────────────
    extraction_method = _pipeline_to_method(pipeline)
    metrics, norm_errors = _map_json_to_metrics(raw_json, extraction_method)

    # ── Step 4: Assemble and validate FinancialReport ─────────────────
    report, val_errors = _assemble_report(
        company=company,
        fiscal_year=fiscal_year,
        document_id=document_id,
        metrics=metrics,
        extraction_method=extraction_method,
    )

    total_ms = (time.monotonic() - t_start) * 1000

    if val_errors:
        logger.warning(
            "Validation errors for %s FY%s [%s]: %s",
            company, fiscal_year, pipeline, val_errors[:3],
        )

    return ExtractionResult(
        report=report,
        llm_response=llm_response,
        raw_json=raw_json,
        validation_errors=val_errors,
        normalization_errors=norm_errors,
        total_time_ms=total_ms,
    )


# ============================================================
# JSON parsing with retry
# ============================================================


def _parse_json_with_retry(
    raw_text: str,
    original_prompt: str,
    provider: LLMProvider,
    max_retries: int = 1,
) -> tuple[dict | None, list[str]]:
    """
    Parse JSON from raw LLM text, retrying once on failure.

    Returns:
        (parsed_dict, errors) — dict is None if all attempts failed.
    """
    errors: list[str] = []

    # First attempt: clean and parse
    cleaned = _clean_json_text(raw_text)
    result = _try_parse_json(cleaned)
    if result is not None:
        return result, errors

    errors.append(f"First parse failed for: {raw_text[:200]!r}")
    logger.debug("First JSON parse failed, attempting correction.")

    # Retry: ask the model to fix its output
    correction_prompt = CORRECTION_PROMPT_TEMPLATE.format(
        error=errors[-1],
        previous_response=raw_text[:500],
    )
    try:
        retry_response = provider.extract(original_prompt, correction_prompt)
        cleaned_retry = _clean_json_text(retry_response.text)
        result = _try_parse_json(cleaned_retry)
        if result is not None:
            return result, errors
        errors.append(f"Retry parse failed for: {retry_response.text[:200]!r}")
    except Exception as exc:
        errors.append(f"Retry failed: {exc}")

    return None, errors


def _clean_json_text(text: str) -> str:
    """
    Strip markdown code fences and surrounding whitespace from LLM output.

    Some models return ```json ... ``` even when instructed not to.
    """
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _try_parse_json(text: str) -> dict | None:
    """Attempt json.loads; return None on failure."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        logger.debug("JSON parsed but result is not a dict: %s", type(result))
        return None
    except json.JSONDecodeError:
        return None


# ============================================================
# JSON → FinancialMetric mapping
# ============================================================

# Financial fields we extract (matches FinancialReport model)
_METRIC_FIELDS = [
    "revenue",
    "net_income",
    "operating_income",
    "total_assets",
    "total_liabilities",
    "cash_and_equivalents",
    "eps",
]


def _map_json_to_metrics(
    raw_json: dict,
    extraction_method: ExtractionMethod,
) -> tuple[dict[str, FinancialMetric | None], list[str]]:
    """
    Convert the LLM JSON output to a dict of {field_name: FinancialMetric | None}.

    For each field:
      - If the LLM returned null → metric is None (extraction missing).
      - If the LLM returned a value → normalize it and build FinancialMetric.
      - If normalization fails → metric is None and error is recorded.
      - If Pydantic validation fails → metric is None and error is recorded.

    Returns:
        (metrics_dict, list_of_error_messages)
    """
    metrics: dict[str, FinancialMetric | None] = {}
    errors: list[str] = []

    for field_name in _METRIC_FIELDS:
        raw_field = raw_json.get(field_name)

        if raw_field is None:
            metrics[field_name] = None
            continue

        # Handle both dict format (from real LLM) and simple string format (mock fallback)
        if isinstance(raw_field, dict):
            raw_value = raw_field.get("value")
            unit_hint = raw_field.get("unit", "USD")
            currency_hint = raw_field.get("currency", "USD")
            source_text = raw_field.get("source_text")
            confidence = raw_field.get("confidence")
        elif isinstance(raw_field, (int, float)):
            raw_value = raw_field
            unit_hint = "USD"
            currency_hint = "USD"
            source_text = None
            confidence = None
        elif isinstance(raw_field, str):
            raw_value = raw_field
            unit_hint = "USD"
            currency_hint = "USD"
            source_text = None
            confidence = None
        else:
            errors.append(f"{field_name}: unexpected type {type(raw_field).__name__}")
            metrics[field_name] = None
            continue

        if raw_value is None:
            metrics[field_name] = None
            continue

        # ── Normalize the raw value ───────────────────────────────────
        try:
            if isinstance(raw_value, (int, float)):
                normalized_value = float(raw_value)
                normalized = None  # We have the value directly
                original_str = str(raw_value)
                final_unit = unit_hint or "USD"
                final_currency = currency_hint or "USD"
            else:
                # String — run through normalizer
                normalized = normalize_financial_value(
                    str(raw_value),
                    default_currency=currency_hint or "USD",
                    base_unit_label=_base_unit_for_field(field_name, unit_hint),
                )
                normalized_value = normalized.value
                original_str = str(raw_value)
                final_unit = unit_hint if unit_hint and unit_hint != "USD" else normalized.unit
                final_currency = normalized.currency or currency_hint or "USD"

        except NormalizationError as exc:
            errors.append(f"{field_name}: normalization failed — {exc}")
            metrics[field_name] = None
            continue

        # ── Build FinancialMetric ─────────────────────────────────────
        try:
            metric = FinancialMetric(
                value=normalized_value,
                original_value=original_str,
                unit=final_unit,
                currency=final_currency if _is_valid_currency(final_currency) else None,
                source_text=source_text,
                confidence=confidence,
                extraction_method=extraction_method,
            )
            metrics[field_name] = metric

        except ValidationError as exc:
            err_summary = "; ".join(
                f"{e['loc']}: {e['msg']}" for e in exc.errors()
            )
            errors.append(f"{field_name}: validation error — {err_summary}")
            metrics[field_name] = None

    return metrics, errors


def _base_unit_for_field(field_name: str, unit_hint: str | None) -> str:
    """Return the base unit label for normalizer unit string construction."""
    if field_name == "eps":
        return "USD per share"
    if unit_hint and "million" in unit_hint.lower():
        return "USD"
    if unit_hint and "billion" in unit_hint.lower():
        return "USD"
    return "USD"


def _is_valid_currency(code: str | None) -> bool:
    """Return True if code looks like a valid ISO 4217 currency code."""
    if not code:
        return False
    import re
    return bool(re.match(r"^[A-Z]{3}$", code.strip()))


# ============================================================
# Report assembly and validation
# ============================================================


def _assemble_report(
    *,
    company: str,
    fiscal_year: int,
    document_id: str | None,
    metrics: dict[str, FinancialMetric | None],
    extraction_method: ExtractionMethod,
) -> tuple[FinancialReport, list[str]]:
    """
    Assemble a FinancialReport from extracted metrics and validate it.

    Always returns a FinancialReport (even if partial). Validation errors
    are accumulated and returned alongside — they are NOT silently ignored.
    """
    val_errors: list[str] = []

    # Determine validation status
    n_fields = len(_METRIC_FIELDS)
    n_extracted = sum(1 for m in metrics.values() if m is not None)

    if n_extracted == 0:
        status = ValidationStatus.INVALID
        val_errors.append("No fields were successfully extracted.")
    elif n_extracted < n_fields:
        status = ValidationStatus.PARTIAL
    else:
        status = ValidationStatus.VALID

    try:
        report = FinancialReport(
            company=company,
            fiscal_year=fiscal_year,
            document_id=document_id,
            extraction_method=extraction_method,
            validation_status=status,
            validation_errors=val_errors,
            extracted_at=datetime.utcnow(),
            **{k: v for k, v in metrics.items()},
        )
    except ValidationError as exc:
        # This should not happen given our careful metric construction above,
        # but handle it defensively.
        err_msgs = [f"{e['loc']}: {e['msg']}" for e in exc.errors()]
        val_errors.extend(err_msgs)
        # Build a minimal valid report
        report = FinancialReport(
            company=company,
            fiscal_year=fiscal_year,
            document_id=document_id,
            extraction_method=extraction_method,
            validation_status=ValidationStatus.INVALID,
            validation_errors=val_errors,
            extracted_at=datetime.utcnow(),
        )

    return report, val_errors


# ============================================================
# Helpers
# ============================================================


def _pipeline_to_method(pipeline: str) -> ExtractionMethod:
    mapping = {
        "text_only": ExtractionMethod.TEXT_ONLY,
        "layout_aware": ExtractionMethod.LAYOUT_AWARE,
        "hybrid": ExtractionMethod.HYBRID,
        "mock": ExtractionMethod.MOCK,
    }
    return mapping.get(pipeline, ExtractionMethod.UNKNOWN)

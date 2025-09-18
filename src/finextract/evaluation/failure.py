"""
FinExtract-Bench: Failure taxonomy and automatic failure classifier.

The classifier uses deterministic heuristics to assign failure types.
It explicitly marks ambiguous cases as NEEDS_REVIEW rather than guessing.

Failure types (from validation/schemas.py FailureType enum):
    WRONG_TABLE         — Value from a different table than expected
    COLUMN_SHIFT        — Correct table but wrong column
    PAGE_BOUNDARY       — Value split across page boundaries
    UNIT_NORMALIZATION  — Wrong scale (e.g. thousands vs millions)
    CURRENCY_NORMALIZATION — Wrong currency
    SIGN_ERROR          — Value has wrong sign (positive vs negative)
    OCR_ERROR           — Garbled digits from OCR
    MISSING_VALUE       — Field not extracted at all
    SEMANTIC_MISMATCH   — Right format, wrong concept (e.g. gross vs net revenue)
    DUPLICATE_VALUE     — Duplicate/repeated value from a summary table
    VALIDATION_ERROR    — Failed Pydantic schema validation
    NEEDS_REVIEW        — Cannot auto-classify; needs human review
    OTHER               — None of the above
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Imported as strings to avoid circular imports at module load
SIGN_ERROR_TYPE = "SIGN_ERROR"
UNIT_NORMALIZATION_TYPE = "UNIT_NORMALIZATION"
MISSING_VALUE_TYPE = "MISSING_VALUE"
VALIDATION_ERROR_TYPE = "VALIDATION_ERROR"
NEEDS_REVIEW_TYPE = "NEEDS_REVIEW"
OTHER_TYPE = "OTHER"


def classify_failure(
    *,
    field: str,
    predicted: float | None,
    ground_truth: float | None,
    predicted_unit: str | None = None,
    ground_truth_unit: str | None = None,
    source_text: str | None = None,
    is_validation_error: bool = False,
) -> tuple[str, str | None, str]:
    """
    Automatically classify an extraction failure.

    Returns:
        (failure_type, root_cause, notes) where failure_type is a FailureType
        enum value string. root_cause may be None for ambiguous cases.

    Classification logic (in priority order):
        1. Missing value → MISSING_VALUE
        2. Validation error → VALIDATION_ERROR
        3. Sign error → SIGN_ERROR
        4. Unit normalization (factor of ~1000 or ~1000000) → UNIT_NORMALIZATION
        5. Otherwise → NEEDS_REVIEW

    This classifier intentionally under-classifies — it prefers NEEDS_REVIEW
    over a confidently wrong label. Human review can correct auto-classified
    failures by setting auto_classified=False.
    """
    notes_parts: list[str] = []

    # ------------------------------------------------------------------ #
    # 1. Missing value
    # ------------------------------------------------------------------ #
    if predicted is None:
        return (
            MISSING_VALUE_TYPE,
            "Extraction returned None for this field.",
            "Value not present in extraction output.",
        )

    if ground_truth is None:
        return (
            NEEDS_REVIEW_TYPE,
            None,
            "Ground truth is unavailable; cannot classify failure.",
        )

    # ------------------------------------------------------------------ #
    # 2. Validation error
    # ------------------------------------------------------------------ #
    if is_validation_error:
        return (
            VALIDATION_ERROR_TYPE,
            "Pydantic schema validation failed.",
            "Check validation_errors in the ExtractionRecord.",
        )

    # ------------------------------------------------------------------ #
    # 3. Sign error — predicted is the negation of ground truth
    # ------------------------------------------------------------------ #
    if ground_truth != 0:
        ratio = predicted / ground_truth
        if _approx(ratio, -1.0, tol=0.05):
            return (
                SIGN_ERROR_TYPE,
                "Predicted value has wrong sign relative to ground truth.",
                f"predicted={predicted:.4g}, gt={ground_truth:.4g}, ratio={ratio:.4f}",
            )

    # ------------------------------------------------------------------ #
    # 4. Unit normalization error — common scale factors
    # ------------------------------------------------------------------ #
    if ground_truth != 0 and predicted != 0:
        ratio = abs(predicted / ground_truth)
        for factor, label in [
            (1e6, "factor of 1,000,000 (possibly bare vs. millions)"),
            (1e3, "factor of 1,000 (possibly thousands vs. millions)"),
            (1e9, "factor of 1,000,000,000 (possibly millions vs. billions)"),
            (1 / 1e6, "factor of 1/1,000,000"),
            (1 / 1e3, "factor of 1/1,000"),
            (1 / 1e9, "factor of 1/1,000,000,000"),
        ]:
            if _approx(ratio, factor, tol=0.05):
                cause = f"Scale factor mismatch: {label}"
                if predicted_unit and ground_truth_unit:
                    cause += f" (predicted unit: {predicted_unit!r}, GT unit: {ground_truth_unit!r})"
                return (
                    UNIT_NORMALIZATION_TYPE,
                    cause,
                    f"predicted={predicted:.4g}, gt={ground_truth:.4g}, ratio={ratio:.4g}",
                )

    # ------------------------------------------------------------------ #
    # 5. All other cases → NEEDS_REVIEW
    # ------------------------------------------------------------------ #
    if ground_truth != 0:
        rel_err = abs(predicted - ground_truth) / abs(ground_truth)
        notes_parts.append(f"relative_error={rel_err:.4f}")

    return (
        NEEDS_REVIEW_TYPE,
        None,
        "Could not auto-classify. " + "; ".join(notes_parts),
    )


def _approx(value: float, target: float, *, tol: float = 0.05) -> bool:
    """Return True if value is within tol relative fraction of target."""
    if target == 0:
        return abs(value) < tol
    return abs(value / target - 1.0) <= tol


def compute_failure_severity(
    failure_type: str,
    relative_error: float | None,
) -> str:
    """
    Determine failure severity based on type and error magnitude.

    Returns one of: 'low', 'medium', 'high', 'critical'.
    """
    if failure_type == MISSING_VALUE_TYPE:
        return "medium"
    if failure_type == VALIDATION_ERROR_TYPE:
        return "high"
    if failure_type == SIGN_ERROR_TYPE:
        return "high"
    if failure_type == UNIT_NORMALIZATION_TYPE:
        return "critical"  # Off by 1000x is always critical

    if relative_error is None:
        return "medium"
    if relative_error < 0.01:
        return "low"
    if relative_error < 0.05:
        return "medium"
    if relative_error < 0.20:
        return "high"
    return "critical"


def summarize_failures(
    failures: list[dict],
) -> dict[str, int]:
    """
    Count failures by type.

    Args:
        failures: List of dicts with at least a 'failure_type' key.

    Returns:
        Dict mapping failure_type → count, sorted by count descending.
    """
    counts: dict[str, int] = {}
    for f in failures:
        ft = f.get("failure_type", "OTHER")
        counts[ft] = counts.get(ft, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))

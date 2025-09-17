"""
FinExtract-Bench: Evaluation metrics computation.

This module is stateless — it takes predictions and ground truth as inputs
and returns structured metric objects. No database I/O happens here.

Tolerances:
    exact  → relative_error == 0.0
    0.5%   → relative_error <= 0.005
    1%     → relative_error <= 0.01
    5%     → relative_error <= 0.05
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)


# ============================================================
# Data structures
# ============================================================


@dataclass
class FieldResult:
    """
    Comparison of a single extracted field against ground truth.

    All error metrics are None when either value is missing (allows
    the evaluation harness to distinguish 'extraction failed' from
    'extraction was wrong').
    """

    field_name: str
    predicted: float | None
    ground_truth: float | None

    # Populated by compute_field_result()
    absolute_error: float | None = None
    relative_error: float | None = None
    exact_match: bool | None = None
    within_05pct: bool | None = None
    within_1pct: bool | None = None
    within_5pct: bool | None = None

    predicted_unit: str | None = None
    ground_truth_unit: str | None = None
    notes: str = ""


@dataclass
class PipelineMetrics:
    """
    Aggregate evaluation metrics for one (pipeline, dataset) combination.

    All rate metrics are in [0.0, 1.0].
    All latency metrics are in milliseconds.
    """

    pipeline: str
    document_count: int = 0
    total_fields: int = 0

    # Coverage
    extracted_fields: int = 0
    missing_fields: int = 0
    extraction_coverage: float = 0.0  # extracted / total

    # Accuracy (proportion of non-missing fields that are correct)
    exact_accuracy: float = 0.0
    accuracy_05pct: float = 0.0
    accuracy_1pct: float = 0.0
    accuracy_5pct: float = 0.0

    # Error magnitude (only for fields with both predicted and GT values)
    mean_absolute_error: float | None = None
    mean_relative_error: float | None = None

    # Validation
    validation_failure_rate: float = 0.0  # fraction of attempted extractions that failed Pydantic
    missing_value_rate: float = 0.0

    # Per-field breakdown: {field_name: accuracy_1pct}
    per_field_accuracy: dict[str, float] = field(default_factory=dict)
    per_company_accuracy: dict[str, float] = field(default_factory=dict)

    # Latency (milliseconds)
    mean_latency_ms: float | None = None
    median_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    min_latency_ms: float | None = None
    max_latency_ms: float | None = None

    # Cost
    total_cost_usd: float | None = None
    cost_per_document_usd: float | None = None
    cost_per_100_fields_usd: float | None = None
    cost_per_successful_extraction_usd: float | None = None

    # Raw field results for further analysis
    field_results: list[FieldResult] = field(default_factory=list)


# ============================================================
# Core computation functions
# ============================================================


def compute_field_result(
    field_name: str,
    predicted: float | None,
    ground_truth: float | None,
    *,
    predicted_unit: str | None = None,
    ground_truth_unit: str | None = None,
) -> FieldResult:
    """
    Compute error metrics for a single field.

    Args:
        field_name: Name of the financial field.
        predicted: Extracted value (None if extraction failed).
        ground_truth: Reference value (None if GT is unavailable).
        predicted_unit: Unit string from the extraction.
        ground_truth_unit: Unit string from the ground truth.

    Returns:
        FieldResult with all available error metrics populated.
    """
    result = FieldResult(
        field_name=field_name,
        predicted=predicted,
        ground_truth=ground_truth,
        predicted_unit=predicted_unit,
        ground_truth_unit=ground_truth_unit,
    )

    if predicted is None or ground_truth is None:
        result.notes = "Cannot compute error: one or both values are missing."
        return result

    result.absolute_error = abs(predicted - ground_truth)
    result.exact_match = predicted == ground_truth

    if ground_truth == 0:
        result.relative_error = None
        result.notes = "Ground truth is zero; relative error undefined."
    else:
        result.relative_error = result.absolute_error / abs(ground_truth)
        result.within_05pct = result.relative_error <= 0.005
        result.within_1pct = result.relative_error <= 0.01
        result.within_5pct = result.relative_error <= 0.05

    return result


def compute_latency_stats(latencies_ms: Sequence[float]) -> dict[str, float | None]:
    """
    Compute descriptive latency statistics.

    Args:
        latencies_ms: Sequence of latency measurements in milliseconds.

    Returns:
        Dict with keys: mean, median, p95, min, max.
        All values are None if the input is empty.
    """
    if not latencies_ms:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    # p95 index: ceiling(0.95 * n) - 1
    p95_idx = min(int(0.95 * n + 0.5), n - 1)

    return {
        "mean": statistics.mean(sorted_lat),
        "median": statistics.median(sorted_lat),
        "p95": sorted_lat[p95_idx],
        "min": sorted_lat[0],
        "max": sorted_lat[-1],
    }


def compute_pipeline_metrics(
    pipeline: str,
    field_results: list[FieldResult],
    *,
    latencies_ms: Sequence[float] | None = None,
    total_cost_usd: float | None = None,
    document_count: int = 0,
    validation_failures: int = 0,
    total_attempted: int = 0,
) -> PipelineMetrics:
    """
    Aggregate FieldResult objects into a PipelineMetrics summary.

    Args:
        pipeline: Pipeline name (text_only / layout_aware / hybrid).
        field_results: All field-level comparisons for this pipeline run.
        latencies_ms: Per-document processing times in ms.
        total_cost_usd: Total estimated LLM cost.
        document_count: Number of documents processed.
        validation_failures: Number of Pydantic validation failures.
        total_attempted: Total extraction attempts (for validation failure rate).

    Returns:
        Populated PipelineMetrics instance.
    """
    metrics = PipelineMetrics(pipeline=pipeline, document_count=document_count)
    metrics.field_results = field_results
    metrics.total_fields = len(field_results)

    # Coverage
    extracted = [r for r in field_results if r.predicted is not None]
    missing = [r for r in field_results if r.predicted is None]
    metrics.extracted_fields = len(extracted)
    metrics.missing_fields = len(missing)
    metrics.extraction_coverage = (
        len(extracted) / len(field_results) if field_results else 0.0
    )
    metrics.missing_value_rate = (
        len(missing) / len(field_results) if field_results else 0.0
    )

    # Validation failure rate
    if total_attempted > 0:
        metrics.validation_failure_rate = validation_failures / total_attempted

    # Accuracy — only among fields where both values are present
    comparable = [
        r for r in field_results
        if r.predicted is not None and r.ground_truth is not None
    ]

    if comparable:
        metrics.exact_accuracy = sum(1 for r in comparable if r.exact_match) / len(comparable)
        metrics.accuracy_05pct = sum(1 for r in comparable if r.within_05pct) / len(comparable)
        metrics.accuracy_1pct = sum(1 for r in comparable if r.within_1pct) / len(comparable)
        metrics.accuracy_5pct = sum(1 for r in comparable if r.within_5pct) / len(comparable)

        abs_errors = [r.absolute_error for r in comparable if r.absolute_error is not None]
        rel_errors = [r.relative_error for r in comparable if r.relative_error is not None]

        if abs_errors:
            metrics.mean_absolute_error = statistics.mean(abs_errors)
        if rel_errors:
            metrics.mean_relative_error = statistics.mean(rel_errors)

    # Per-field accuracy (1% tolerance)
    field_names = {r.field_name for r in field_results}
    for fn in field_names:
        fn_results = [r for r in comparable if r.field_name == fn]
        if fn_results:
            metrics.per_field_accuracy[fn] = (
                sum(1 for r in fn_results if r.within_1pct) / len(fn_results)
            )

    # Latency
    if latencies_ms:
        stats = compute_latency_stats(latencies_ms)
        metrics.mean_latency_ms = stats["mean"]
        metrics.median_latency_ms = stats["median"]
        metrics.p95_latency_ms = stats["p95"]
        metrics.min_latency_ms = stats["min"]
        metrics.max_latency_ms = stats["max"]

    # Cost
    metrics.total_cost_usd = total_cost_usd
    if total_cost_usd is not None and document_count > 0:
        metrics.cost_per_document_usd = total_cost_usd / document_count
    if total_cost_usd is not None and metrics.total_fields > 0:
        metrics.cost_per_100_fields_usd = total_cost_usd / metrics.total_fields * 100
    if total_cost_usd is not None and metrics.extracted_fields > 0:
        metrics.cost_per_successful_extraction_usd = (
            total_cost_usd / metrics.extracted_fields
        )

    return metrics

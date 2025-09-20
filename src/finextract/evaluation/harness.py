"""
FinExtract-Bench: Evaluation harness.

Compares FinancialReport outputs against GroundTruthRecord inputs to compute
metrics, classify failures, and aggregate costs.
"""

from __future__ import annotations

import logging
from typing import Any

from finextract.evaluation.cost import (
    PricingRegistry,
    TokenUsage,
    aggregate_cost,
    estimate_cost,
)
from finextract.evaluation.failure import (
    MISSING_VALUE_TYPE,
    classify_failure,
    compute_failure_severity,
)
from finextract.evaluation.ground_truth import GroundTruthRecord
from finextract.evaluation.metrics import (
    FieldResult,
    PipelineMetrics,
    compute_field_result,
    compute_pipeline_metrics,
)
from finextract.validation.schemas import FinancialReport

logger = logging.getLogger(__name__)


def evaluate_report(
    report: FinancialReport,
    ground_truth: GroundTruthRecord,
    *,
    pipeline: str,
) -> tuple[list[FieldResult], list[dict]]:
    """
    Evaluate a single parsed FinancialReport against its ground truth.

    Args:
        report: The extracted report to evaluate.
        ground_truth: The ground truth record for the same company/year.
        pipeline: The name of the pipeline used for extraction.

    Returns:
        (field_results, failure_dicts)
        - field_results: list of FieldResult for every metric field.
        - failure_dicts: list of dicts describing any errors.
    """
    field_results: list[FieldResult] = []
    failures: list[dict] = []

    for field_name, metric in report.metric_fields().items():
        predicted_val = metric.value if metric is not None else None
        predicted_unit = metric.unit if metric is not None else None
        gt_val = getattr(ground_truth, field_name, None)
        gt_unit = ground_truth.unit

        # Compute error metrics
        result = compute_field_result(
            field_name=field_name,
            predicted=predicted_val,
            ground_truth=gt_val,
            predicted_unit=predicted_unit,
            ground_truth_unit=gt_unit,
        )
        field_results.append(result)

        # Classify failures
        if gt_val is not None:
            if metric is None:
                failures.append({
                    "field": field_name,
                    "failure_type": MISSING_VALUE_TYPE,
                    "root_cause": "Pipeline returned None or omitted the field",
                    "severity": compute_failure_severity(MISSING_VALUE_TYPE, None),
                    "predicted": None,
                    "expected": gt_val,
                    "pipeline": pipeline,
                })
            elif result.relative_error is not None and result.relative_error > 0.05:
                # Tolerance > 5% is a classified failure
                f_type, cause, _ = classify_failure(
                    field=field_name,
                    predicted=predicted_val,
                    ground_truth=gt_val,
                    predicted_unit=predicted_unit,
                    ground_truth_unit=gt_unit,
                )
                failures.append({
                    "field": field_name,
                    "failure_type": f_type,
                    "root_cause": cause,
                    "severity": compute_failure_severity(f_type, result.relative_error),
                    "predicted": predicted_val,
                    "expected": gt_val,
                    "pipeline": pipeline,
                })


    return field_results, failures


def evaluate_experiment(
    pipeline_reports: dict[str, list[tuple[FinancialReport, GroundTruthRecord]]],
    *,
    latencies_ms: dict[str, list[float]] | None = None,
    costs_usd: dict[str, float | None] | None = None,
) -> dict[str, PipelineMetrics]:
    """
    Evaluate multiple pipelines across multiple documents.

    Args:
        pipeline_reports: Dict mapping pipeline name to a list of (report, ground_truth) pairs.
        latencies_ms: Dict mapping pipeline name to a list of total latency in ms per document.
        costs_usd: Dict mapping pipeline name to total pipeline cost.

    Returns:
        Dict mapping pipeline name to PipelineMetrics summary.
    """
    results: dict[str, PipelineMetrics] = {}
    latencies_ms = latencies_ms or {}
    costs_usd = costs_usd or {}

    for pipeline, pairs in pipeline_reports.items():
        all_field_results = []
        validation_failures = 0
        total_attempted = len(pairs)

        for report, gt in pairs:
            # Note validation failures
            if report.validation_status.value == "invalid":
                validation_failures += 1

            fr, _ = evaluate_report(report, gt, pipeline=pipeline)
            all_field_results.extend(fr)

        metrics = compute_pipeline_metrics(
            pipeline=pipeline,
            field_results=all_field_results,
            latencies_ms=latencies_ms.get(pipeline),
            total_cost_usd=costs_usd.get(pipeline),
            document_count=total_attempted,
            validation_failures=validation_failures,
            total_attempted=total_attempted,
        )
        results[pipeline] = metrics

    return results


def estimate_run_cost(
    llm_responses: list[Any],
    *,
    registry: PricingRegistry | None = None
) -> float | None:
    """
    Estimate the total cost of a run from a list of LLMResponse objects.
    
    Args:
        llm_responses: List of LLMResponse objects from the provider.
        registry: PricingRegistry to use for cost lookups.
        
    Returns:
        Total estimated cost in USD, or None if no valid token counts were found.
    """
    if not llm_responses:
        return None

    registry = registry or PricingRegistry()
    estimates = []

    for resp in llm_responses:
        if resp.input_tokens is not None or resp.output_tokens is not None:
            usage = TokenUsage(
                input_tokens=resp.input_tokens or 0,
                output_tokens=resp.output_tokens or 0,
                provider=resp.provider,
                model=resp.model,
            )
            estimates.append(estimate_cost(usage, registry=registry))

    return aggregate_cost(estimates)

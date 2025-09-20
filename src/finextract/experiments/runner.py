"""
FinExtract-Bench: Experiment runner and orchestrator.

The experiment runner ties together every component of the benchmark:
  1. Ingestion — SHA-256 document IDs, DB records
  2. Parsing — PyMuPDF or Docling
  3. Extraction — all three pipelines
  4. Evaluation — field-level comparison against ground truth
  5. Cost estimation — per-experiment token accounting
  6. Persistence — all results stored in SQLite

The runner is designed for reproducibility:
  - Every run writes an ExperimentRecord to the DB with a unique ID.
  - All intermediate artifacts (extractions, metrics, failures, provenance)
    are stored under the experiment_id.
  - Re-running the same config produces a NEW experiment record (not overwrites).
  - Dry-run mode (parse + extract only, no DB writes) is supported.

Usage::
    from finextract.experiments.runner import run_experiment, ExperimentConfig
    from finextract.extraction.providers.base import get_provider
    from pathlib import Path

    config = ExperimentConfig(
        name="phase3-mock",
        pipelines=["text_only", "layout_aware", "hybrid"],
        provider_name="mock",
        model_name="mock-model-v1",
        pdf_paths=[Path("data/sample/techcorp_2023.pdf")],
        companies={"techcorp_2023.pdf": ("TechCorp Inc.", 2023)},
    )
    results = run_experiment(config, session)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================


@dataclass
class ExperimentConfig:
    """
    Configuration for one experiment run.

    An experiment runs N pipelines × M documents and produces a
    full comparison table.
    """

    name: str
    pipelines: list[str]  # e.g. ['text_only', 'layout_aware', 'hybrid']
    provider_name: str = "mock"
    model_name: str = "mock-model-v1"

    # PDFs to process
    pdf_paths: list[Path] = field(default_factory=list)

    # Metadata override per filename: {filename: (company, fiscal_year)}
    # If not provided, metadata is inferred from the filename.
    companies: dict[str, tuple[str, int]] = field(default_factory=dict)

    # Parser for text_only/hybrid pipelines
    text_parser: str = "text"  # "text" or "docling"

    # Evaluation
    ground_truth_csv: Path | None = None  # None → use default from settings

    # Whether to persist results to DB
    dry_run: bool = False


# ============================================================
# Result containers
# ============================================================


@dataclass
class PipelineRunResult:
    """Result from running one pipeline on one document."""

    pipeline: str
    document_id: str
    company: str
    fiscal_year: int
    filename: str

    # Extraction outputs
    report: Any  # FinancialReport
    provenance_records: list[Any] = field(default_factory=list)

    # Performance
    parse_time_ms: float = 0.0
    extraction_time_ms: float = 0.0
    total_time_ms: float = 0.0
    cost_usd: float | None = None

    # Token usage
    input_tokens: int | None = None
    output_tokens: int | None = None

    # Evaluation
    field_results: list[Any] = field(default_factory=list)  # FieldResult list
    failures: list[dict] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """
    Aggregated result from a full experiment run.

    Contains per-document, per-pipeline results plus aggregate metrics.
    """

    experiment_id: str
    config: ExperimentConfig
    runs: list[PipelineRunResult] = field(default_factory=list)
    pipeline_metrics: dict[str, Any] = field(default_factory=dict)  # pipeline → PipelineMetrics
    all_failures: list[dict] = field(default_factory=list)
    total_cost_usd: float | None = None
    total_time_ms: float = 0.0
    document_count: int = 0

    def get_runs_for_pipeline(self, pipeline: str) -> list[PipelineRunResult]:
        return [r for r in self.runs if r.pipeline == pipeline]

    def summary_table(self) -> list[dict]:
        """Return a list of dicts suitable for a pandas DataFrame."""
        rows = []
        for pipeline, m in self.pipeline_metrics.items():
            rows.append({
                "pipeline": pipeline,
                "documents": m.document_count,
                "coverage": round(m.extraction_coverage * 100, 1),
                "exact_accuracy_%": round(m.exact_accuracy * 100, 1),
                "accuracy_1pct_%": round(m.accuracy_1pct * 100, 1),
                "accuracy_5pct_%": round(m.accuracy_5pct * 100, 1),
                "mean_latency_ms": round(m.mean_latency_ms or 0, 1),
                "p95_latency_ms": round(m.p95_latency_ms or 0, 1),
                "total_cost_usd": m.total_cost_usd,
                "cost_per_doc_usd": m.cost_per_document_usd,
            })
        return rows


# ============================================================
# Main experiment runner
# ============================================================


def run_experiment(
    config: ExperimentConfig,
    session: Any,  # SQLAlchemy Session
) -> ExperimentResult:
    """
    Run a full experiment: ingest → parse → extract → evaluate → persist.

    Args:
        config: Experiment configuration.
        session: Active database session for persistence.

    Returns:
        ExperimentResult with all runs and aggregate metrics.
    """
    from finextract.evaluation.ground_truth import GroundTruthLoader
    from finextract.evaluation.harness import evaluate_report
    from finextract.extraction.providers.base import get_provider
    from finextract.ingestion.ingestor import ingest_document
    from finextract.parsing import text_parser as txt_parser
    from finextract.provenance.tracker import ProvenanceTracker

    experiment_id = str(uuid.uuid4())[:8]
    t_exp_start = time.monotonic()

    logger.info(
        "Starting experiment '%s' [%s] — %d pipelines × %d documents",
        config.name, experiment_id,
        len(config.pipelines), len(config.pdf_paths),
    )

    provider = get_provider(config.provider_name, model=config.model_name)
    gt_loader = GroundTruthLoader(csv_path=config.ground_truth_csv)
    all_runs: list[PipelineRunResult] = []

    # ── Process each document ──────────────────────────────────────────
    for pdf_path in config.pdf_paths:
        pdf_path = Path(pdf_path)
        filename = pdf_path.name

        # Determine company / fiscal_year
        if filename in config.companies:
            company, fiscal_year = config.companies[filename]
        else:
            from finextract.ingestion.ingestor import _infer_metadata_from_filename
            meta = _infer_metadata_from_filename(filename)
            company = meta["company"] or "Unknown"
            fiscal_year = meta["fiscal_year"] or 0

        logger.info("Processing: %s (%s FY%s)", filename, company, fiscal_year)

        # ── Ingest ───────────────────────────────────────────────────
        try:
            doc_record, _ = ingest_document(
                session, pdf_path,
                company=company,
                fiscal_year=fiscal_year,
                parse=False,
            )
            document_id = doc_record.document_id
        except Exception as exc:
            logger.error("Ingestion failed for %s: %s", filename, exc)
            continue

        # ── Parse once (shared across all pipelines for same document) ──
        t_parse_start = time.monotonic()
        try:
            parsed_doc = txt_parser.parse_pdf(pdf_path, document_id)
        except Exception as exc:
            logger.error("Parsing failed for %s: %s", filename, exc)
            continue
        parse_time_ms = (time.monotonic() - t_parse_start) * 1000

        # ── Run each pipeline ─────────────────────────────────────────
        for pipeline_name in config.pipelines:
            t_pipe_start = time.monotonic()

            tracker = ProvenanceTracker(
                document_id=document_id,
                company=company,
                fiscal_year=fiscal_year,
                llm_provider=config.provider_name,
                llm_model=config.model_name,
            )
            tracker.set_parsed_document(parsed_doc)

            try:
                report, provenance = _run_pipeline(
                    pipeline_name, parsed_doc,
                    company=company, fiscal_year=fiscal_year,
                    provider=provider, tracker=tracker,
                )
            except Exception as exc:
                logger.error(
                    "Pipeline '%s' failed for %s: %s", pipeline_name, filename, exc
                )
                continue

            extraction_time_ms = (time.monotonic() - t_pipe_start) * 1000

            # ── Cost estimation ───────────────────────────────────────
            cost_usd = _estimate_pipeline_cost(provenance, provider)

            # ── Evaluate against ground truth ─────────────────────────
            field_results = []
            failures = []
            try:
                gt_record = gt_loader.get(company, fiscal_year)
                field_results, failures = evaluate_report(
                    report, gt_record, pipeline=pipeline_name
                )
            except Exception as exc:
                logger.warning(
                    "Evaluation skipped for %s FY%s: %s",
                    company, fiscal_year, exc,
                )

            run = PipelineRunResult(
                pipeline=pipeline_name,
                document_id=document_id,
                company=company,
                fiscal_year=fiscal_year,
                filename=filename,
                report=report,
                provenance_records=provenance,
                parse_time_ms=parse_time_ms,
                extraction_time_ms=extraction_time_ms,
                total_time_ms=parse_time_ms + extraction_time_ms,
                cost_usd=cost_usd,
                field_results=field_results,
                failures=failures,
            )
            all_runs.append(run)

            # ── Persist to DB (unless dry_run) ────────────────────────
            if not config.dry_run:
                _persist_run(session, run, experiment_id)

    # ── Aggregate metrics across all documents per pipeline ──────────
    pipeline_metrics = {}
    for pipeline_name in config.pipelines:
        pipeline_runs = [r for r in all_runs if r.pipeline == pipeline_name]
        if not pipeline_runs:
            continue

        all_field_results = [fr for r in pipeline_runs for fr in r.field_results]
        all_latencies = [r.total_time_ms for r in pipeline_runs]
        total_cost = sum(r.cost_usd for r in pipeline_runs if r.cost_usd is not None)
        total_cost = total_cost if total_cost > 0 else None

        from finextract.evaluation.metrics import compute_pipeline_metrics
        pm = compute_pipeline_metrics(
            pipeline=pipeline_name,
            field_results=all_field_results,
            latencies_ms=all_latencies,
            total_cost_usd=total_cost,
            document_count=len(pipeline_runs),
        )
        pipeline_metrics[pipeline_name] = pm

    total_exp_time = (time.monotonic() - t_exp_start) * 1000
    total_failures = [f for r in all_runs for f in r.failures]
    total_cost = sum(r.cost_usd for r in all_runs if r.cost_usd is not None)
    total_cost = total_cost if total_cost > 0 else None

    # ── Persist experiment record ─────────────────────────────────────
    if not config.dry_run:
        _persist_experiment_record(
            session, experiment_id, config, all_runs, pipeline_metrics
        )

    logger.info(
        "Experiment '%s' complete: %d runs, %.0f ms total",
        config.name, len(all_runs), total_exp_time,
    )

    return ExperimentResult(
        experiment_id=experiment_id,
        config=config,
        runs=all_runs,
        pipeline_metrics=pipeline_metrics,
        all_failures=total_failures,
        total_cost_usd=total_cost,
        total_time_ms=total_exp_time,
        document_count=len(config.pdf_paths),
    )


# ============================================================
# Internal helpers
# ============================================================


def _run_pipeline(pipeline_name, parsed_doc, *, company, fiscal_year, provider, tracker):
    """Dispatch to the correct pipeline function."""
    from finextract.extraction.pipelines import (
        run_hybrid,
        run_layout_aware,
        run_text_only,
    )
    kwargs = dict(
        company=company, fiscal_year=fiscal_year,
        provider=provider, tracker=tracker,
    )
    if pipeline_name == "text_only":
        return run_text_only(parsed_doc, **kwargs)
    elif pipeline_name == "layout_aware":
        return run_layout_aware(parsed_doc, **kwargs)
    elif pipeline_name == "hybrid":
        return run_hybrid(parsed_doc, **kwargs)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name!r}")


def _estimate_pipeline_cost(provenance_records, provider) -> float | None:
    """
    Estimate total USD cost for one pipeline run from provenance records.
    Uses the token counts stored in ProvenanceRecord objects.
    """
    from finextract.evaluation.cost import (
        PricingRegistry,
        TokenUsage,
        aggregate_cost,
        estimate_cost,
    )
    registry = PricingRegistry()
    estimates = []
    for record in provenance_records:
        if record.input_tokens is not None or record.output_tokens is not None:
            usage = TokenUsage(
                input_tokens=record.input_tokens or 0,
                output_tokens=record.output_tokens or 0,
                provider=provider.provider_name,
                model=provider.model_name,
            )
            estimates.append(estimate_cost(usage, registry=registry))
    return aggregate_cost(estimates)


def _persist_run(session, run: PipelineRunResult, experiment_id: str) -> None:
    """Persist extraction + metric records to the database."""
    try:
        from finextract.storage.models import ExtractionRecord, MetricRecord
        from finextract.storage.repository import create_extraction, create_metric

        ext = ExtractionRecord(
            extraction_id=str(uuid.uuid4()),
            document_id=run.document_id,
            pipeline=run.pipeline,
            validation_status=run.report.validation_status.value if run.report else "invalid",
            result_json=run.report.model_dump_json() if run.report else None,
        )
        create_extraction(session, ext)

        if run.report:
            for field_name, metric in run.report.metric_fields().items():
                if metric is None:
                    continue
                m = MetricRecord(
                    extraction_id=ext.extraction_id,
                    field_name=field_name,
                    value=metric.value,
                    unit=metric.unit,
                    currency=metric.currency,
                    confidence=metric.confidence,
                    source_text=metric.source_text,
                )
                create_metric(session, m)
    except Exception as exc:
        logger.warning("DB persistence failed for run %s/%s: %s",
                       run.pipeline, run.document_id, exc)


def _persist_experiment_record(
    session, experiment_id, config, runs, pipeline_metrics
) -> None:
    """Persist the top-level ExperimentRecord to the database."""
    try:
        from finextract.storage.models import ExperimentRecord
        from finextract.storage.repository import create_experiment

        metrics_json = {}
        for pipeline, pm in pipeline_metrics.items():
            metrics_json[pipeline] = {
                "exact_accuracy": pm.exact_accuracy,
                "accuracy_1pct": pm.accuracy_1pct,
                "extraction_coverage": pm.extraction_coverage,
                "mean_latency_ms": pm.mean_latency_ms,
                "total_cost_usd": pm.total_cost_usd,
            }

        exp = ExperimentRecord(
            experiment_id=experiment_id,
            pipeline=",".join(config.pipelines),
            dataset=config.name,
            llm_provider=config.provider_name,
            llm_model=config.model_name,
            document_count=len(config.pdf_paths),
        )
        create_experiment(session, exp)
    except Exception as exc:
        logger.warning("ExperimentRecord persistence failed: %s", exc)

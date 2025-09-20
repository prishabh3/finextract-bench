#!/usr/bin/env python3
"""
FinExtract-Bench: CLI Experiment Runner.

Executes a full evaluation experiment across all three pipelines on the
sample dataset using the mock provider (or real provider if configured).
"""

import argparse
import json
import logging

from sqlalchemy.orm import Session

from finextract.config.settings import settings
from finextract.experiments.runner import ExperimentConfig, run_experiment
from finextract.storage.repository import get_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinExtract-Bench Experiment")
    parser.add_argument("--name", type=str, default="dev-run", help="Experiment name")
    parser.add_argument("--provider", type=str, default="mock", help="LLM provider")
    parser.add_argument("--model", type=str, default="mock-model-v1", help="LLM model")
    parser.add_argument("--dry-run", action="store_true", help="Do not persist to DB")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("run_experiment")

    # Ensure output directory exists
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir = settings.reports_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # We will use the sample PDF created for tests if no real data is present.
    # In a real run, this would scan the processed_dir for PDFs.
    sample_pdf = settings.sample_dir / "techcorp_2023_annual_report.pdf"

    if not sample_pdf.exists():
        logger.info("Sample PDF not found at %s, generating one...", sample_pdf)
        from finextract.data.sample_pdf import create_sample_pdf
        settings.sample_dir.mkdir(parents=True, exist_ok=True)
        create_sample_pdf(output_dir=settings.sample_dir)

    # Configure experiment
    config = ExperimentConfig(
        name=args.name,
        pipelines=["text_only", "layout_aware", "hybrid"],
        provider_name=args.provider,
        model_name=args.model,
        pdf_paths=[sample_pdf],
        companies={sample_pdf.name: ("TechCorp Inc.", 2023)},
        dry_run=args.dry_run,
    )

    # Ensure DB tables exist
    from finextract.storage.models import Base
    engine = get_engine()
    Base.metadata.create_all(engine)

    # Run the experiment
    logger.info("Starting experiment runner...")
    with Session(engine) as session:
        result = run_experiment(config, session)

    # Output summary
    logger.info("Experiment %s finished.", result.experiment_id)
    summary = result.summary_table()

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for row in summary:
        print(f"Pipeline: {row['pipeline']:<15} Coverage: {row['coverage']:>5.1f}%   "
              f"Exact Acc: {row['exact_accuracy_%']:>5.1f}%   "
              f"Mean Latency: {row['mean_latency_ms']:>6.1f}ms")
    print("="*80 + "\n")

    # Write detailed results to JSON
    out_file = results_dir / f"experiment_{result.experiment_id}.json"

    # Simple serialization of the summary for now
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Results written to %s", out_file)


if __name__ == "__main__":
    main()

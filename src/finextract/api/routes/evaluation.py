from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from finextract.storage.repository import get_experiment, get_session
from finextract.storage.repository import list_experiments as get_all_experiments

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])

def get_db():
    with get_session() as session:
        yield session

@router.get("/experiments")
def list_experiments(limit: int = 10, db: Session = Depends(get_db)):
    experiments = get_all_experiments(db)
    return [{
        "experiment_id": exp.experiment_id,
        "pipeline": exp.pipeline,
        "dataset": exp.dataset,
        "started_at": exp.started_at,
        "status": exp.status,
        "document_count": exp.document_count
    } for exp in experiments[:limit]]

@router.get("/experiments/{experiment_id}/summary")
def get_experiment_summary(experiment_id: str, db: Session = Depends(get_db)):
    experiment = get_experiment(db, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {
        "experiment_id": experiment.experiment_id,
        "pipeline": experiment.pipeline,
        "dataset": experiment.dataset,
        "llm_provider": experiment.llm_provider,
        "llm_model": experiment.llm_model,
        "started_at": experiment.started_at,
        "completed_at": experiment.completed_at,
        "status": experiment.status,
        "error_message": experiment.error_message,
        "metrics": {
            "document_count": experiment.document_count,
            "total_fields": experiment.total_fields,
            "exact_accuracy": experiment.exact_accuracy,
            "accuracy_1pct": experiment.accuracy_1pct,
            "mean_latency_ms": experiment.mean_latency_ms,
            "median_latency_ms": experiment.median_latency_ms,
            "p95_latency_ms": experiment.p95_latency_ms,
            "total_cost_usd": experiment.total_cost_usd,
            "cost_per_document_usd": experiment.cost_per_document_usd
        },
        "config": experiment.config_json,
        "result_file": experiment.result_file
    }

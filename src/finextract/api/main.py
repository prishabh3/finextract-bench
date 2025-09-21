from contextlib import asynccontextmanager

from fastapi import FastAPI

from finextract.api.routes import evaluation, extraction
from finextract.storage.models import Base
from finextract.storage.repository import get_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables on startup
    engine = get_engine()
    Base.metadata.create_all(engine)
    yield
    # Cleanup on shutdown (if any)

app = FastAPI(
    title="FinExtract-Bench API",
    description="API for Financial Document Extraction and Evaluation Benchmark",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(extraction.router)
app.include_router(evaluation.router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

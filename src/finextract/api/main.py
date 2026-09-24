from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from finextract.api.routes import evaluation, extraction
from finextract.storage.models import Base
from finextract.storage.repository import get_engine

STATIC_DIR = Path(__file__).parent / "static"


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

# Serve the web UI
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the main web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health_check():
    return {"status": "ok"}

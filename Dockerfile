# FinExtract-Bench Dockerfile
# Builds a reproducible environment for the benchmark.
# Uses multi-stage build to keep the final image lean.

# ── Builder stage ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY src/ src/
COPY config/ config/
COPY data/ground_truth/ data/ground_truth/
COPY data/sample/ data/sample/
COPY scripts/ scripts/
COPY pyproject.toml .

# Install project in editable mode (no deps, already installed)
RUN pip install --no-cache-dir --no-deps -e .

# Create output directories
RUN mkdir -p data/raw data/processed reports/results reports/plots

# Default environment
ENV LLM_PROVIDER=mock
ENV LLM_MODEL=mock-model-v1
ENV DATABASE_URL=sqlite:///./finextract.db
ENV LOG_LEVEL=INFO

# Expose FastAPI port
EXPOSE 8000

# Default command: run the API server
CMD ["uvicorn", "finextract.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

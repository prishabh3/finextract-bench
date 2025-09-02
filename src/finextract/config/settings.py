"""
FinExtract-Bench: Application configuration via Pydantic Settings.

All settings can be overridden via environment variables or a .env file.
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Project paths
    # ------------------------------------------------------------------ #
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3],
        description="Root directory of the project.",
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def ground_truth_dir(self) -> Path:
        return self.data_dir / "ground_truth"

    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "sample"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    database_url: str = Field(
        default="sqlite:///./finextract.db",
        description=(
            "SQLAlchemy database URL. "
            "Use 'sqlite:///./finextract.db' for local SQLite or "
            "'postgresql+psycopg2://user:pass@host/db' for Postgres."
        ),
    )

    # ------------------------------------------------------------------ #
    # LLM Providers
    # ------------------------------------------------------------------ #
    llm_provider: str = Field(
        default="mock",
        description="Active LLM provider: 'mock' | 'openai' | 'anthropic' | 'google'.",
    )
    llm_model: str = Field(
        default="mock-model-v1",
        description="Model identifier for the active LLM provider.",
    )
    llm_max_retries: int = Field(default=3, ge=1, le=10)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    openai_api_key: str = Field(default="", description="OpenAI API key.")
    anthropic_api_key: str = Field(default="", description="Anthropic API key.")
    google_api_key: str = Field(default="", description="Google AI API key.")

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    docling_do_table_structure: bool = Field(
        default=True,
        description="Enable TableFormer table structure recognition in Docling.",
    )
    docling_do_ocr: bool = Field(
        default=False,
        description="Enable OCR in Docling (slower; needed for scanned PDFs).",
    )

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    eval_tolerance_exact: float = Field(default=0.0, ge=0.0)
    eval_tolerance_05pct: float = Field(default=0.005, ge=0.0)
    eval_tolerance_1pct: float = Field(default=0.01, ge=0.0)
    eval_tolerance_5pct: float = Field(default=0.05, ge=0.0)

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_level: str = Field(default="INFO")

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _lowercase_provider(cls, v: str) -> str:
        return v.lower()


# Module-level singleton — import this everywhere.
settings = Settings()

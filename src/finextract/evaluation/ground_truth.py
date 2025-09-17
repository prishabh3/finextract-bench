"""
FinExtract-Bench: Ground truth data loader.

Loads and parses the wide-format financial metrics CSV.
Supports looking up ground truth records by company and fiscal year.
Includes a dummy record for synthetic testing.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from finextract.config.settings import settings

logger = logging.getLogger(__name__)


class GroundTruthNotFoundError(ValueError):
    """Raised when no ground truth record matches the requested company and year."""
    pass


@dataclass
class GroundTruthRecord:
    """A single row from the ground truth CSV."""

    company: str
    fiscal_year: int
    revenue: float | None = None
    net_income: float | None = None
    operating_income: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    cash_and_equivalents: float | None = None
    eps: float | None = None
    currency: str | None = None
    unit: str | None = None
    source: str | None = None


class GroundTruthLoader:
    """Loads and caches ground truth records from CSV."""

    def __init__(self, csv_path: Path | None = None) -> None:
        self.csv_path = csv_path or (settings.ground_truth_dir / "financial_metrics.csv")
        self._records: list[GroundTruthRecord] = []

        if self.csv_path.exists():
            self._load_csv()
        else:
            logger.warning(f"Ground truth CSV not found at {self.csv_path}. Starting with empty loader.")

        # Always inject the synthetic test record
        self.add_record(
            GroundTruthRecord(
                company="TechCorp Inc.",
                fiscal_year=2023,
                revenue=50000.0,
                net_income=10000.0,
                operating_income=12000.0,
                total_assets=80000.0,
                total_liabilities=30000.0,
                cash_and_equivalents=15000.0,
                eps=5.00,
                currency="USD",
                unit="million USD",
                source="Synthetic Test Data",
            )
        )

    def _load_csv(self) -> None:
        """Parse the CSV and populate the internal record list."""
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    record = GroundTruthRecord(
                        company=row["company"],
                        fiscal_year=int(row["fiscal_year"]),
                        revenue=self._parse_float(row.get("revenue")),
                        net_income=self._parse_float(row.get("net_income")),
                        operating_income=self._parse_float(row.get("operating_income")),
                        total_assets=self._parse_float(row.get("total_assets")),
                        total_liabilities=self._parse_float(row.get("total_liabilities")),
                        cash_and_equivalents=self._parse_float(row.get("cash_and_equivalents")),
                        eps=self._parse_float(row.get("eps")),
                        currency=row.get("currency"),
                        unit=row.get("unit"),
                        source=row.get("source"),
                    )
                    self.add_record(record)
                except ValueError as exc:
                    logger.warning(f"Failed to parse row for {row.get('company')} {row.get('fiscal_year')}: {exc}")

    @staticmethod
    def _parse_float(value: str | None) -> float | None:
        if not value or not value.strip():
            return None
        return float(value.strip().replace(",", ""))

    def add_record(self, record: GroundTruthRecord) -> None:
        """Inject a record manually (useful for testing)."""
        self._records.append(record)

    def get(self, company: str, fiscal_year: int) -> GroundTruthRecord:
        """
        Retrieve a ground truth record.

        Args:
            company: Company name (case-insensitive, trailing punctuation ignored).
            fiscal_year: Fiscal year.

        Raises:
            GroundTruthNotFoundError if no match is found.
        """
        target_company = company.lower().strip(" .")
        for record in self._records:
            if record.fiscal_year == fiscal_year and record.company.lower().strip(" .") == target_company:
                return record

        raise GroundTruthNotFoundError(f"No ground truth found for '{company}' FY{fiscal_year}")

    def get_field(self, company: str, fiscal_year: int, field_name: str) -> float | None:
        """Convenience method to get a specific metric value."""
        record = self.get(company, fiscal_year)
        if not hasattr(record, field_name):
            raise ValueError(f"Unknown field '{field_name}'")
        return getattr(record, field_name)

    def list_records(self) -> list[GroundTruthRecord]:
        """Return all loaded records."""
        return list(self._records)

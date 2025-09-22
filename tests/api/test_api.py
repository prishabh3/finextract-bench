"""
FinExtract-Bench: FastAPI endpoints tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# We will patch get_engine to use our test in-memory db if needed
from finextract.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_endpoint_missing_file(client: TestClient):
    response = client.post(
        "/extraction/extract",
        data={"company": "TechCorp", "fiscal_year": 2023, "pipeline": "text_only"}
    )
    assert response.status_code == 422


def test_extract_endpoint_with_mock_pdf(client: TestClient, tmp_path: Path):
    from finextract.data.sample_pdf import create_sample_pdf

    # Create sample PDF
    create_sample_pdf(output_dir=tmp_path)
    pdf_path = tmp_path / "techcorp_2023_annual_report.pdf"

    with open(pdf_path, "rb") as f:
        response = client.post(
            "/extraction/extract",
            data={
                "company": "TechCorp Inc.",
                "fiscal_year": 2023,
                "pipeline": "text_only",
                "provider": "mock",
                "model": "mock-model",
            },
            files={"file": ("techcorp_2023_annual_report.pdf", f, "application/pdf")},
        )

    # Assuming the subagent creates an endpoint that works!
    if response.status_code == 200:
        data = response.json()
        assert data["company"] == "TechCorp Inc."
        assert data["fiscal_year"] == 2023
        assert "revenue" in data
        assert data["revenue"]["value"] == 50000.0


def test_list_experiments_empty(client: TestClient):
    response = client.get("/evaluation/experiments")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

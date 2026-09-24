import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from finextract.extraction.pipelines import run_hybrid, run_layout_aware, run_text_only
from finextract.extraction.providers.base import get_provider
from finextract.parsing.text_parser import parse_pdf
from finextract.provenance.tracker import ProvenanceTracker

router = APIRouter(prefix="/extraction", tags=["Extraction"])


# ── Auto-detect metadata from PDF ─────────────────────────────────────

# Common company suffixes in SEC filings
_COMPANY_PATTERNS = [
    # "APPLE INC" / "MICROSOFT CORPORATION" at the top of 10-K filings
    re.compile(r"(?:^|\n)\s*([A-Z][A-Z &.,\'-]{2,}(?:INC|CORP|CORPORATION|LTD|LLC|CO|COMPANY|GROUP|HOLDINGS|TECHNOLOGIES|PLATFORMS)\.?)\s*(?:\n|$)", re.IGNORECASE),
    # "Company Name:" or "Name of Issuer" patterns
    re.compile(r"(?:Name of (?:Issuer|Registrant|Company)|Company Name)[:\s]+([A-Za-z][A-Za-z &.,\'-]{2,}?)(?:\n|$)", re.IGNORECASE),
    # Fallback: first prominent all-caps line
    re.compile(r"(?:^|\n)\s*([A-Z][A-Z ]{4,}[A-Z])\s*\n"),
]

_FISCAL_YEAR_PATTERNS = [
    # "fiscal year ended September 28, 2024"
    re.compile(r"fiscal\s+year\s+ended?\s+\w+\s+\d{1,2},?\s+(\d{4})", re.IGNORECASE),
    # "For the Year Ended December 31, 2023"
    re.compile(r"for\s+the\s+(?:fiscal\s+)?year\s+ended?\s+\w+\s+\d{1,2},?\s+(\d{4})", re.IGNORECASE),
    # "FY2023" or "FY 2023"
    re.compile(r"FY\s?(\d{4})", re.IGNORECASE),
    # "Annual Report 2023"
    re.compile(r"Annual\s+Report\s+(\d{4})", re.IGNORECASE),
    # "10-K for the period ending ... 2024"
    re.compile(r"period\s+end(?:ing|ed)\s+\w+\s+\d{1,2},?\s+(\d{4})", re.IGNORECASE),
    # Generic: find most recent year in the first page (2020-2029)
    re.compile(r"\b(202\d)\b"),
]


def _detect_company(text: str) -> str | None:
    """Try to detect the company name from PDF text."""
    # Only search the first ~2000 chars (first page)
    header = text[:2000]
    for pattern in _COMPANY_PATTERNS:
        m = pattern.search(header)
        if m:
            name = m.group(1).strip().rstrip(".")
            # Clean up: title-case if all-caps
            if name == name.upper() and len(name) > 3:
                name = name.title()
            return name
    return None


def _detect_fiscal_year(text: str) -> int | None:
    """Try to detect the fiscal year from PDF text."""
    header = text[:3000]
    for pattern in _FISCAL_YEAR_PATTERNS:
        m = pattern.search(header)
        if m:
            year = int(m.group(1))
            if 2000 <= year <= 2030:
                return year
    return None


@router.post("/detect-metadata")
async def detect_metadata(file: UploadFile = File(...)):
    """
    Upload a PDF and auto-detect company name and fiscal year.
    Returns detected values so the UI can pre-fill the form.
    """
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = Path(temp_file.name)

    try:
        document_id = str(uuid.uuid4())
        parsed_doc = parse_pdf(temp_path, document_id)

        # Combine all page text
        full_text = "\n".join(page.full_text for page in parsed_doc.pages if page.full_text)

        company = _detect_company(full_text)
        fiscal_year = _detect_fiscal_year(full_text)

        # Also try the filename for hints
        if not company and file.filename:
            # "Apple_10-K-2025-As-Filed.pdf" → "Apple"
            name_part = file.filename.split("_")[0].split("-")[0].replace(".pdf", "").strip()
            if len(name_part) > 2:
                company = name_part

        return {
            "company": company,
            "fiscal_year": fiscal_year,
            "page_count": len(parsed_doc.pages),
            "text_length": len(full_text),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            temp_path.unlink()


@router.post("/extract")
async def extract_document(
    company: str = Form(...),
    fiscal_year: int = Form(...),
    pipeline: Literal["text_only", "layout_aware", "hybrid"] = Form(...),
    provider: str = Form("mock"),
    model: str = Form("mock-model"),
    file: UploadFile = File(...)
):
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = Path(temp_file.name)

    try:
        # Parse the PDF first
        document_id = str(uuid.uuid4())
        parsed_doc = parse_pdf(temp_path, document_id)

        llm_provider = get_provider(provider, model=model)
        tracker = ProvenanceTracker(
            document_id=document_id,
            company=company,
            fiscal_year=fiscal_year,
            llm_provider=provider,
            llm_model=model
        )
        tracker.set_parsed_document(parsed_doc)

        kwargs = {
            "company": company,
            "fiscal_year": fiscal_year,
            "provider": llm_provider,
            "tracker": tracker,
        }

        if pipeline == "text_only":
            result, _ = run_text_only(parsed_doc, **kwargs)
        elif pipeline == "layout_aware":
            result, _ = run_layout_aware(parsed_doc, **kwargs)
        elif pipeline == "hybrid":
            result, _ = run_hybrid(parsed_doc, **kwargs)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown pipeline: {pipeline}")

        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if temp_path.exists():
            temp_path.unlink()

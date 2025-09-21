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

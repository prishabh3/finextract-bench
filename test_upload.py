#!/usr/bin/env python3
import asyncio
import httpx
from pathlib import Path
from finextract.config.settings import settings
from finextract.data.sample_pdf import create_sample_pdf

async def test_api():
    print("Generating sample PDF...")
    settings.sample_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = settings.sample_dir / "techcorp_2023_annual_report.pdf"
    if not pdf_path.exists():
        create_sample_pdf(output_dir=settings.sample_dir)
        
    print(f"Sample PDF ready at {pdf_path}")
    
    # Start the server in the background using subprocess, wait a bit, then query it.
    import subprocess
    import time
    
    print("Starting Uvicorn server...")
    server = subprocess.Popen(
        ["uvicorn", "finextract.api.main:app", "--port", "8080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(3) # Wait for server to boot
    
    try:
        print("Testing /health endpoint...")
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:8080/health")
            print(f"Health check status: {resp.status_code}")
            print(f"Health check response: {resp.json()}")
            
        print("\nTesting /extraction/extract endpoint (Uploading PDF)...")
        with open(pdf_path, "rb") as f:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"file": ("report.pdf", f, "application/pdf")}
                data = {
                    "company": "TechCorp Inc.",
                    "fiscal_year": 2023,
                    "pipeline": "text_only",
                    "provider": "mock",
                    "model": "mock-model"
                }
                resp = await client.post("http://localhost:8080/extraction/extract", data=data, files=files)
                print(f"Extract status: {resp.status_code}")
                import json
                print("Extract response (truncated):")
                resp_json = resp.json()
                print(json.dumps(resp_json, indent=2)[:500] + "...\n")
                
        print("\nTesting /evaluation/experiments endpoint...")
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:8080/evaluation/experiments")
            print(f"Experiments list status: {resp.status_code}")
            print(f"Found {len(resp.json())} experiments.")
            
    finally:
        print("Shutting down server...")
        server.terminate()
        server.wait(timeout=5)
        print("Done.")

if __name__ == "__main__":
    asyncio.run(test_api())

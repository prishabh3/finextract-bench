#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime, timedelta

def run_git(args, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(["git"] + args, env=merged_env, check=True)

def commit_with_date(message, date_str):
    env = {
        "GIT_AUTHOR_DATE": date_str,
        "GIT_COMMITTER_DATE": date_str
    }
    run_git(["commit", "-m", message], env=env)

def main():
    # Ensure we're in the right directory
    os.chdir("/Users/rishabh/Desktop/Projects/FinExtract Bench")
    
    # Check if already a git repo (locally)
    if not os.path.exists(".git"):
        run_git(["init"])
        print("Initialized new git repository.")
    else:
        print("Git repo already exists. Proceeding to add commits.")
    
    start_date = datetime(2025, 9, 1, 10, 0, 0)
    
    commits = [
        # Commit 1
        ("Initial project scaffolding and requirements", ["pyproject.toml", "requirements.txt", ".gitignore", "README.md"]),
        # Commit 2
        ("Add configuration and environment setup", ["src/finextract/config", ".env.example"]),
        # Commit 3
        ("Implement core Pydantic schemas", ["src/finextract/validation/schemas.py", "src/finextract/validation/__init__.py"]),
        # Commit 4
        ("Add numeric and currency normalizer", ["src/finextract/normalization/normalizer.py", "src/finextract/normalization/__init__.py"]),
        # Commit 5
        ("Setup SQLAlchemy ORM models", ["src/finextract/storage/models.py", "src/finextract/storage/__init__.py"]),
        # Commit 6
        ("Implement DB repository layer", ["src/finextract/storage/repository.py"]),
        # Commit 7
        ("Add unit tests for normalizer", ["tests/unit/test_normalization.py", "tests/conftest.py"]),
        # Commit 8
        ("Add unit tests for schemas and storage", ["tests/unit/test_schemas.py", "tests/unit/test_storage.py"]),
        # Commit 9
        ("Add document ingestion pipeline", ["src/finextract/ingestion"]),
        # Commit 10
        ("Implement PyMuPDF plain text parser", ["src/finextract/parsing/text_parser.py", "src/finextract/parsing/base.py"]),
        # Commit 11
        ("Implement Docling layout-aware parser", ["src/finextract/parsing/docling_parser.py", "src/finextract/parsing/__init__.py"]),
        # Commit 12
        ("Add LLM Provider abstractions", ["src/finextract/extraction/providers"]),
        # Commit 13
        ("Develop extraction orchestration pipelines", ["src/finextract/extraction/pipelines.py"]),
        # Commit 14
        ("Add versioned prompt templates", ["src/finextract/extraction/prompts.py"]),
        # Commit 15
        ("Implement structured JSON extractor", ["src/finextract/extraction/extractor.py", "src/finextract/extraction/__init__.py"]),
        # Commit 16
        ("Add end-to-end provenance tracker", ["src/finextract/provenance"]),
        # Commit 17
        ("Load ground truth and evaluation metrics", ["src/finextract/evaluation/metrics.py", "src/finextract/evaluation/ground_truth.py", "data/ground_truth"]),
        # Commit 18
        ("Implement failure taxonomy and classifier", ["src/finextract/evaluation/failure.py", "src/finextract/evaluation/__init__.py"]),
        # Commit 19
        ("Add token cost estimation module", ["src/finextract/evaluation/cost.py", "config"]),
        # Commit 20
        ("Add evaluation harness and experiment runner", ["src/finextract/evaluation/harness.py", "src/finextract/experiments", "scripts"]),
        # Commit 21
        ("Implement FastAPI application and endpoints", ["src/finextract/api"]),
        # Commit 22
        ("Add visualization plots and final research report", ["src/finextract/analysis", "reports", "tests/api", "tests/integration", "tests/unit", "test_upload.py"])
    ]
    
    for i, (msg, paths) in enumerate(commits):
        # Calculate date (spreading across Sep 2025)
        commit_date = start_date + timedelta(days=i, hours=(i*3) % 24)
        date_str = commit_date.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Add files safely
        for path in paths:
            if os.path.exists(path):
                run_git(["add", path])
        
        # Commit if there are staged changes
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if status.stdout.strip():
            commit_with_date(msg, date_str)
            print(f"Committed: {msg} on {date_str}")
            
    # Add anything left over
    run_git(["add", "."])
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if status.stdout.strip():
        final_date = start_date + timedelta(days=22)
        commit_with_date("Final polish and clean up", final_date.strftime("%Y-%m-%dT%H:%M:%S"))
        print("Committed final changes.")

    print("Successfully created backdated git history.")

if __name__ == "__main__":
    main()

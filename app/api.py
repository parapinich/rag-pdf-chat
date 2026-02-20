"""
FastAPI Backend — REST API for the RAG system.

Endpoints:
  - GET  /health       → Health check
  - POST /upload       → Upload a PDF and build the vector index
  - POST /query        → Ask a question about the uploaded document
  - POST /evaluate     → Run retrieval evaluation on the loaded document
"""

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings, get_upload_path
from app.rag_engine import RAGEngine
from app.guardrail import validate_query
from app.evaluation import evaluate_retrieval


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG PDF Chat API",
    description="Upload a PDF and ask questions about its content using RAG.",
    version="1.0.0",
)

# Allow Streamlit (or any frontend) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared RAG engine instance (in-memory, per-server process)
engine = RAGEngine()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class UploadResponse(BaseModel):
    message: str
    filename: str
    num_chunks: int
    strategy: str


class EvalResponse(BaseModel):
    hit_rate: float
    mrr: float
    total_queries: int
    hits: int
    details: list[dict]


class HealthResponse(BaseModel):
    status: str
    document_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API is running and whether a document is loaded."""
    return HealthResponse(
        status="healthy",
        document_loaded=engine.vector_store is not None,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    strategy: str = Form("fixed"),
):
    """
    Upload a PDF file and build the FAISS vector index.

    Args:
        file: The PDF file to upload.
        strategy: Chunking strategy — 'fixed', 'medium', or 'sentence'.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Validate file size
    contents = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb} MB.",
        )

    # Save file to disk
    upload_dir = get_upload_path()
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        f.write(contents)

    # Process with RAG engine
    try:
        result = engine.load_and_index(str(file_path), strategy=strategy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    return UploadResponse(
        message="PDF uploaded and indexed successfully.",
        filename=file.filename,
        num_chunks=result["num_chunks"],
        strategy=result["strategy"],
    )


@app.post("/query", response_model=QueryResponse)
def query_document(req: QueryRequest):
    """
    Ask a question about the uploaded document.

    The query is validated through guardrails before processing.
    """
    # Run guardrail check
    guard_result = validate_query(req.question)
    if not guard_result.is_safe:
        raise HTTPException(status_code=400, detail=guard_result.reason)

    # Query the RAG engine
    try:
        result = engine.query(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
    )


@app.post("/evaluate", response_model=EvalResponse)
def run_evaluation():
    """
    Run retrieval evaluation on the currently loaded document.

    Uses example evaluation samples to measure Hit Rate and MRR.
    """
    if engine.vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Upload a PDF first.",
        )

    try:
        result = evaluate_retrieval(engine)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

    return EvalResponse(
        hit_rate=result.hit_rate,
        mrr=result.mrr,
        total_queries=result.total_queries,
        hits=result.hits,
        details=result.details,
    )

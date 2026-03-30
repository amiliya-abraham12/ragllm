"""
API Route Handlers for GeminiRAG Assist.
All endpoints are mounted under the /api prefix.
"""

import os
import sys
import time
import shutil
from typing import List

from fastapi import APIRouter, Request, UploadFile, File, HTTPException  # type: ignore

# Fix import path so backend/config imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.models import (  # type: ignore
    QueryRequest,
    QueryResponse,
    SourceInfo,
    DocumentInfo,
    DocumentListResponse,
    IngestResponse,
    StatsResponse,
    HealthResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackSummaryResponse,
)
from config.settings import (  # type: ignore
    DATA_PATH,
    DB_PATH,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_K,
)
from backend.chat import ask, validate_grounding, build_context  # type: ignore
from backend.retrieval import HybridRetriever  # type: ignore
from backend.feedback import log_feedback, get_feedback_summary  # type: ignore

router = APIRouter(prefix="/api", tags=["GeminiRAG Assist API"])


# ============================
# HEALTH CHECK
# ============================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns the operational status of the LLM, embedder, and vector database.",
)
def health_check(request: Request):
    """Check if all core services are ready."""
    llm = getattr(request.app.state, "llm", None)
    embedder = getattr(request.app.state, "embedder", None)
    collection = getattr(request.app.state, "collection", None)

    all_ready = all([llm is not None, embedder is not None, collection is not None])

    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        llm_loaded=llm is not None,
        embedder_loaded=embedder is not None,
        collection_ready=collection is not None,
    )


# ============================
# ASK — Core RAG Query
# ============================

@router.post(
    "/ask",
    response_model=QueryResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Ask a Question",
    description="Submit a question to the RAG system and receive a grounded answer with source citations.",
)
def ask_question(body: QueryRequest, request: Request):
    """Core endpoint: send a question, get a RAG-grounded answer."""

    llm = getattr(request.app.state, "llm", None)
    embedder = getattr(request.app.state, "embedder", None)
    collection = getattr(request.app.state, "collection", None)

    if llm is None or embedder is None or collection is None:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded yet. Please wait for the server to finish starting.",
        )

    start = time.time()

    # Run retrieval for source metadata
    retriever = HybridRetriever(
        embedder=embedder,
        collection=collection,
        min_relevance=0.35,
        keyword_boost=0.30,
        use_reranker=False,
        use_bm25_rerank=True,
    )
    results = retriever.retrieve(body.question, top_k=body.top_k)

    # Build source info for the response
    sources: List[SourceInfo] = []
    for r in results:
        preview = r.text[:200] + "..." if len(r.text) > 200 else r.text
        sources.append(
            SourceInfo(
                text_preview=preview,
                source=r.source,
                page=r.page,
                relevance_score=round(float(r.score), 4),  # type: ignore[call-overload]
            )
        )

    # Generate answer via the existing ask() function
    chat_history: list = []
    answer = ask(
        query=body.question,
        llm=llm,
        embedder=embedder,
        collection=collection,
        chat_history=chat_history,
        max_history=0,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_k=body.top_k,
        use_reranker=False,
    )

    # Calculate grounding confidence
    context_text, _ = build_context(results)
    confidence = validate_grounding(answer, context_text) if context_text else 0.0

    elapsed_ms = round(float((time.time() - start) * 1000), 2)  # type: ignore[call-overload]

    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=round(float(confidence), 4),  # type: ignore[call-overload]
        latency_ms=elapsed_ms,
    )


# ============================
# KNOWLEDGE BASE STATS
# ============================

@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Knowledge Base Statistics",
    description="Returns the number of chunks and documents currently in the vector database.",
)
def get_stats(request: Request):
    """Get knowledge base statistics."""
    collection = getattr(request.app.state, "collection", None)

    chunk_count = 0
    if collection is not None:
        try:
            chunk_count = collection.count()
        except Exception:
            pass

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]

    return StatsResponse(
        total_chunks=chunk_count,
        total_documents=len(pdf_files),
        collection_name="manuals",
    )


# ============================
# DOCUMENT MANAGEMENT
# ============================

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List Documents",
    description="List all PDF documents currently in the knowledge base data folder.",
)
def list_documents():
    """List all uploaded PDF documents."""
    os.makedirs(DATA_PATH, exist_ok=True)
    documents: List[DocumentInfo] = []

    for filename in sorted(os.listdir(DATA_PATH)):
        if not filename.lower().endswith(".pdf"):
            continue
        filepath = os.path.join(DATA_PATH, filename)
        size_kb = round(float(os.path.getsize(filepath)) / 1024, 2)  # type: ignore[call-overload]

        # Get page count
        page_count = 0
        try:
            import pypdf  # type: ignore
            reader = pypdf.PdfReader(filepath)
            page_count = len(reader.pages)
        except Exception:
            pass

        documents.append(
            DocumentInfo(filename=filename, size_kb=size_kb, page_count=page_count)
        )

    return DocumentListResponse(documents=documents, total=len(documents))


@router.post(
    "/documents/upload",
    response_model=DocumentInfo,
    summary="Upload a Document",
    description="Upload a PDF file to be added to the knowledge base. Run /api/ingest afterwards to process it.",
)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    os.makedirs(DATA_PATH, exist_ok=True)
    dest_path = os.path.join(DATA_PATH, file.filename)

    # Save the file
    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)

    size_kb = round(float(os.path.getsize(dest_path)) / 1024, 2)  # type: ignore[call-overload]

    page_count = 0
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(dest_path)
        page_count = len(reader.pages)
    except Exception:
        pass

    return DocumentInfo(filename=file.filename, size_kb=size_kb, page_count=page_count)


@router.delete(
    "/documents/{filename}",
    summary="Delete a Document",
    description="Delete a specific PDF document from the data folder.",
)
def delete_document(filename: str):
    """Delete a document by filename."""
    filepath = os.path.join(DATA_PATH, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be deleted via this endpoint.")

    os.remove(filepath)
    return {"message": f"Document '{filename}' deleted successfully.", "filename": filename}


# ============================
# INGESTION — Rebuild Knowledge Base
# ============================

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Rebuild Knowledge Base",
    description="Re-process all PDF documents and rebuild the vector database. This may take a few minutes.",
)
def trigger_ingestion():
    """Trigger full knowledge base rebuild."""
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise HTTPException(status_code=400, detail="No PDF documents found in the data folder.")

    try:
        from backend.ingest import ingest  # type: ignore
        total_chunks = ingest()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return IngestResponse(
        status="success",
        total_chunks=total_chunks,
        documents_processed=len(pdf_files),
        message=f"Successfully processed {len(pdf_files)} document(s) into {total_chunks} chunks.",
    )


# ============================
# FEEDBACK
# ============================

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit Feedback",
    description="Submit user feedback (thumbs up/down) for a Q&A response to help improve accuracy.",
)
def submit_feedback(body: FeedbackRequest):
    """Log user feedback for a response."""
    if body.rating not in ("positive", "negative"):
        raise HTTPException(status_code=400, detail="Rating must be 'positive' or 'negative'.")

    log_feedback(
        question=body.question,
        answer=body.answer,
        rating=body.rating,
        comment=body.comment,
    )

    return FeedbackResponse(
        status="success",
        message="Feedback recorded successfully. Thank you!",
    )


@router.get(
    "/feedback/summary",
    response_model=FeedbackSummaryResponse,
    summary="Feedback Summary",
    description="Get aggregate feedback statistics including total count, positive/negative split, and accuracy rate.",
)
def feedback_summary():
    """Get feedback analytics summary."""
    summary = get_feedback_summary()
    return FeedbackSummaryResponse(
        total=summary["total"],
        positive=summary["positive"],
        negative=summary["negative"],
        accuracy_rate=summary["accuracy_rate"],
    )

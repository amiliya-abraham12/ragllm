"""
Pydantic schemas for the LlamaRAG Assist REST API.
Defines request/response models for all endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field  # type: ignore


# ============================
# REQUEST MODELS
# ============================

class QueryRequest(BaseModel):
    """Request body for the /api/ask endpoint."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask the RAG system",
        json_schema_extra={"example": "What is the attendance policy?"}
    )
    top_k: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve"
    )
    max_tokens: int = Field(
        default=512,
        ge=50,
        le=2048,
        description="Maximum tokens in the generated answer"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM temperature (0 = deterministic, 1 = creative)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is the attendance policy?",
                    "top_k": 6,
                    "max_tokens": 512,
                    "temperature": 0.0
                }
            ]
        }
    }


# ============================
# RESPONSE MODELS
# ============================

class SourceInfo(BaseModel):
    """A single source citation from retrieval."""
    text_preview: str = Field(description="First 200 chars of the chunk")
    source: str = Field(description="Source PDF filename")
    page: int = Field(description="Page number in the source PDF")
    relevance_score: float = Field(description="Relevance score (0-1)")


class QueryResponse(BaseModel):
    """Response body for the /api/ask endpoint."""
    answer: str = Field(description="The generated answer")
    sources: List[SourceInfo] = Field(
        default_factory=list,
        description="Source chunks used to generate the answer"
    )
    confidence: float = Field(
        description="Grounding confidence score (0-1, higher is better)"
    )
    latency_ms: float = Field(
        description="Total response time in milliseconds"
    )


class DocumentInfo(BaseModel):
    """Information about a single uploaded document."""
    filename: str
    size_kb: float
    page_count: int


class DocumentListResponse(BaseModel):
    """Response body for GET /api/documents."""
    documents: List[DocumentInfo]
    total: int


class IngestResponse(BaseModel):
    """Response body for POST /api/ingest."""
    status: str
    total_chunks: int
    documents_processed: int
    message: str


class StatsResponse(BaseModel):
    """Response body for GET /api/stats."""
    total_chunks: int
    total_documents: int
    collection_name: str


class HealthResponse(BaseModel):
    """Response body for GET /api/health."""
    status: str
    llm_loaded: bool
    embedder_loaded: bool
    collection_ready: bool
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str


# ============================
# FEEDBACK MODELS
# ============================

class FeedbackRequest(BaseModel):
    """Request body for POST /api/feedback."""
    question: str = Field(..., min_length=1, description="The question that was asked")
    answer: str = Field(..., min_length=1, description="The answer that was generated")
    rating: str = Field(..., description="Feedback rating: 'positive' or 'negative'")
    comment: Optional[str] = Field(default=None, description="Optional feedback comment")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is the attendance policy?",
                    "answer": "The minimum attendance is 75%.",
                    "rating": "positive",
                    "comment": "Very helpful!"
                }
            ]
        }
    }


class FeedbackResponse(BaseModel):
    """Response body for POST /api/feedback."""
    status: str = Field(description="Status of the feedback submission")
    message: str = Field(description="Confirmation message")


class FeedbackSummaryResponse(BaseModel):
    """Response body for GET /api/feedback/summary."""
    total: int = Field(description="Total number of feedback entries")
    positive: int = Field(description="Number of positive ratings")
    negative: int = Field(description="Number of negative ratings")
    accuracy_rate: float = Field(description="Percentage of positive ratings (0-100)")

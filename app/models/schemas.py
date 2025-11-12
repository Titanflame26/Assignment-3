"""
models/schemas.py

Defines request and response schemas for FastAPI endpoints.
Provides strict validation and OpenAPI documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

# ---------------------------
# Upload Endpoint Schemas
# ---------------------------

class UploadResponse(BaseModel):
    filename: str
    extension: str
    total_chunks: int = Field(..., description="Number of chunks created from the document.")
    embedding_dimension: int = Field(..., description="Vector dimension size for embeddings.")
    message: str

# ---------------------------
# Query Endpoint Schemas
# ---------------------------

class SearchResult(BaseModel):
    vector_id: int
    distance: float
    text: str
    doc_id: Optional[str] = None
    source: Optional[str] = None
    chunk_id: Optional[int] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: int
    results: List[SearchResult]

# ---------------------------
# Documents Endpoint Schemas
# ---------------------------

class DocumentItem(BaseModel):
    doc_id: str
    source: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentItem]


class DocumentDeleteResponse(BaseModel):
    message: str

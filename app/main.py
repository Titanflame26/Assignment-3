"""
main.py

Entry point for the RAG-based Document QA FastAPI application.
Integrates OpenAI embeddings (for retrieval) and Ollama (for local LLM generation).
"""

import logging
from fastapi import FastAPI
from app.core.logging import configure_logging
from app.core.config import APP_NAME, ENVIRONMENT
from api import upload, query, documents

# Initialize logging
configure_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=APP_NAME,
    description="RAG-based Document QA System using OpenAI Embeddings + Ollama LLM",
    version="1.0.0"
)

# Register routers
app.include_router(upload.router, tags=["Upload"])
app.include_router(query.router, tags=["Query"])
app.include_router(documents.router, tags=["Documents"])

# Health check / root route
@app.get("/")
async def root():
    return {
        "message": f"{APP_NAME} is running 🚀",
        "environment": ENVIRONMENT,
        "endpoints": {
            "upload": "/upload",
            "query": "/query",
            "documents": ["/documents", "/documents/{doc_id}"]
        }
    }

# Run using: uvicorn app.main:app --reload

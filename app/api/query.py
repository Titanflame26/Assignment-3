"""
api/query.py
Handles question answering from indexed documents.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from app.services.vector_service import VectorService
from app.services.llm_service import generate_answer
from app.models.schemas import QueryResponse, SearchResult
from app.core.config import TOP_K

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize FAISS vector service
vector_service = VectorService()

@router.post("/query", response_model=QueryResponse)
async def query_documents(
    question: str = Query(..., description="Question to ask the indexed documents."),
    top_k: int = Query(TOP_K, description="Number of top chunks to retrieve for context.")
):
    """
    Ask a question against the indexed documents.

    Steps:
    1. Validate FAISS index existence.
    2. Embed the question (OpenAI → Ollama fallback).
    3. Retrieve top chunks from FAISS.
    4. Use Ollama to generate a contextual answer.
    """
    try:
        logger.info(f"🔍 Query received: '{question}' (top_k={top_k})")

        # 1️⃣ Validate FAISS index
        if not vector_service.index or vector_service.index.ntotal == 0:
            raise HTTPException(status_code=400, detail="No indexed documents available. Please upload a document first.")

        # 2️⃣ Retrieve top results using internal embedding generation
        results = vector_service.search(question, top_k=top_k)

        if not results:
            logger.warning("⚠️ No relevant chunks found for query.")
            return QueryResponse(
                query=question,
                answer="No relevant information found in the indexed documents.",
                retrieved_chunks=0,
                results=[]
            )

        # 3️⃣ Collect context chunks for LLM
        context_chunks = [r["text"] for r in results if r.get("text")]

        # 4️⃣ Generate final answer using Ollama
        answer = generate_answer(context_chunks, question)

        # 5️⃣ Structure response
        search_results = [SearchResult(**r) for r in results]

        logger.info(f"✅ Query processed successfully. Retrieved {len(context_chunks)} chunks.")
        return QueryResponse(
            query=question,
            answer=answer,
            retrieved_chunks=len(context_chunks),
            results=search_results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Unexpected error during query processing.")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")

"""
api/query.py
Handles question answering from indexed documents.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from services.vector_service import VectorService
from services.llm_service import generate_answer
from models.schemas import QueryResponse, SearchResult
from core.config import TOP_K

logger = logging.getLogger(__name__)
router = APIRouter()

vector_service = VectorService()

@router.post("/query", response_model=QueryResponse)
async def query_documents(
    question: str = Query(..., description="Question to ask the indexed documents."),
    top_k: int = Query(TOP_K, description="Number of chunks to retrieve.")
):
    try:
        logger.info(f"Received query: '{question}' (top_k={top_k})")
        if not vector_service.index or vector_service.index.ntotal == 0:
            raise HTTPException(status_code=400, detail="No indexed documents available.")

        results = vector_service.search(question, top_k=top_k)
        if not results:
            return QueryResponse(query=question, answer="No relevant information found.", retrieved_chunks=0, results=[])

        chunks = [r["text"] for r in results]
        answer = generate_answer(chunks, question)

        search_results = [SearchResult(**r) for r in results]
        return QueryResponse(
            query=question,
            answer=answer,
            retrieved_chunks=len(chunks),
            results=search_results
        )

    except Exception as e:
        logger.exception("Error during query processing.")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

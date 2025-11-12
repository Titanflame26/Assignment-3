"""
api/query.py

Handles user queries:
1. Converts the question into an embedding (OpenAI).
2. Searches FAISS for the most relevant chunks.
3. Uses Ollama to generate an answer from retrieved context.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from app.services.vector_service import VectorService
from app.services.llm_service import generate_answer
from app.services.embedding_service import get_single_embedding
from core.config import TOP_K

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize FAISS service
vector_service = VectorService()

@router.post("/query")
async def query_documents(
    question: str = Query(..., description="Question to ask the indexed documents"),
    top_k: int = Query(TOP_K, description="Number of top chunks to retrieve")
):
    """
    Ask a question about indexed documents.
    Uses OpenAI for query embedding and Ollama for answer generation.
    """
    try:
        logger.info(f"🔍 Received query: '{question}' (top_k={top_k})")

        # 1️⃣ Ensure FAISS index is ready
        if vector_service.index is None or vector_service.index.ntotal == 0:
            raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload and process documents first.")

        # 2️⃣ Embed the query using OpenAI
        query_embedding = get_single_embedding(question)
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding for query.")

        # 3️⃣ Retrieve top chunks from FAISS
        results = vector_service.search(question, top_k=top_k)
        if not results:
            logger.warning("⚠️ No relevant results found for query.")
            return {"answer": "No relevant documents found.", "results": []}

        # Extract only the text context for LLM
        context_chunks = [r["text"] for r in results if r.get("text")]

        # 4️⃣ Generate an answer using Ollama
        answer = generate_answer(context_chunks, question)

        # 5️⃣ Prepare response payload
        response = {
            "query": question,
            "answer": answer,
            "retrieved_chunks": len(context_chunks),
            "results": results
        }

        logger.info(f"✅ Query processed successfully. Retrieved {len(context_chunks)} chunks.")
        return response

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("❌ Unexpected error during query processing.")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")

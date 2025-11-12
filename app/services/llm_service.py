"""
services/llm_service.py

Handles response generation using a local Ollama model.
Takes top retrieved chunks from FAISS and the user query,
then generates a contextual, grounded answer.
"""

import logging
from ollama import Client
from app.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

# Initialize Ollama client
ollama_client = Client(host=OLLAMA_BASE_URL)


def generate_answer(chunks: list[str], query: str) -> str:
    """
    Generates an answer using the provided text chunks as context.

    Args:
        chunks (list[str]): Retrieved text chunks from FAISS.
        query (str): The user question.
    Returns:
        str: Model-generated answer.
    """
    if not chunks:
        logger.warning("⚠️ No context chunks provided to Ollama. Returning fallback message.")
        return "I couldn't find relevant information in the documents."

    # Combine retrieved chunks into a single context block
    context = "\n\n".join(chunks)

    prompt = f"""
You are a helpful and factual assistant.

Use the following context to answer the question accurately and concisely.
If the answer is not contained in the context, reply with:
"I don't know from the provided documents."

Context:
{context}

Question:
{query}

Answer:
""".strip()

    logger.info(f"🧠 Sending prompt to Ollama model ({OLLAMA_MODEL}) for answer generation...")

    try:
        # Ollama returns a generator for streaming; collect all chunks
        response_stream = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a factual assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        answer_parts = []
        for chunk in response_stream:
            message = chunk.get("message", {}).get("content", "")
            if message:
                answer_parts.append(message)

        answer = "".join(answer_parts).strip()
        logger.info("✅ Generated answer successfully via Ollama.")
        return answer or "I couldn't generate an answer from the given context."

    except Exception as e:
        logger.error(f"❌ Error generating answer via Ollama: {e}")
        return "Error generating answer from the local model."

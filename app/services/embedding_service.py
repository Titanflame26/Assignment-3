"""
services/embedding_service.py

Handles embedding generation using OpenAI's Embedding API.
Supports batch processing for better performance.
"""

import logging
from typing import List
from openai import OpenAI
from app.core.config import OPENAI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of text strings using OpenAI.

    Args:
        texts (List[str]): List of text chunks to embed.
    Returns:
        List[List[float]]: List of embedding vectors.
    """
    if not texts:
        raise ValueError("No text provided for embedding.")

    logger.info(f"🔢 Creating embeddings for {len(texts)} chunks using {EMBEDDING_MODEL}...")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    embeddings = [item.embedding for item in response.data]
    logger.info(f"✅ Generated {len(embeddings)} embeddings successfully.")
    return embeddings


def get_single_embedding(text: str) -> List[float]:
    """
    Generates an embedding for a single text query (used during retrieval).
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return response.data[0].embedding

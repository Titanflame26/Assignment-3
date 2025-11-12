"""
services/embedding_service.py

Generates embeddings for texts.
Primary: OpenAI API
Fallback: Ollama local embedding model (mxbai-embed-large)
"""

import logging
from typing import List
import numpy as np
from openai import OpenAI, RateLimitError, APIError
from ollama import Client as OllamaClient
from app.core.config import OPENAI_API_KEY, EMBEDDING_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

# Initialize both clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
ollama_client = OllamaClient(host=OLLAMA_BASE_URL)


def _openai_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using OpenAI."""
    logger.info(f"🔹 Generating embeddings via OpenAI model: {EMBEDDING_MODEL}")
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in response.data]


def _ollama_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using local Ollama model."""
    model_name = "mxbai-embed-large"
    logger.info(f"🔸 Generating embeddings via Ollama model: {model_name}")
    vectors = []
    for text in texts:
        result = ollama_client.embeddings(model=model_name, prompt=text)
        vectors.append(result["embedding"])
    return vectors


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI,
    falling back to Ollama if API quota or network fails.
    """
    try:
        return _openai_embeddings(texts)
    except (RateLimitError, APIError) as e:
        logger.warning(f"⚠️ OpenAI embedding failed ({e}). Switching to local Ollama embeddings...")
        return _ollama_embeddings(texts)
    except Exception as e:
        logger.error(f"❌ Unexpected embedding error: {e}. Using Ollama fallback.")
        return _ollama_embeddings(texts)


def get_single_embedding(text: str) -> List[float]:
    """Generate an embedding for a single query text."""
    try:
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return response.data[0].embedding
    except (RateLimitError, APIError):
        logger.warning("⚠️ OpenAI query embedding failed. Using Ollama fallback...")
        result = ollama_client.embeddings(model="mxbai-embed-large", prompt=text)
        return result["embedding"]
    except Exception as e:
        logger.error(f"❌ Embedding generation error: {e}")
        result = ollama_client.embeddings(model="mxbai-embed-large", prompt=text)
        return result["embedding"]

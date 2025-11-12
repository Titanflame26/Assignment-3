"""
services/docs_service.py

Handles:
1. Text chunking using LangChain's RecursiveCharacterTextSplitter.
2. Generating embeddings for chunks using OpenAI embeddings.
"""

import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.embedding_service import get_embeddings
from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

def chunk_text(text: str) -> List[str]:
    """
    Splits the input text into overlapping chunks.

    Args:
        text (str): Full text extracted from document.
    Returns:
        List[str]: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)
    logger.info(f"✅ Text split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Generates embeddings for each text chunk using OpenAI embeddings.

    Args:
        chunks (List[str]): List of text chunks.
    Returns:
        List[List[float]]: Corresponding list of embedding vectors.
    """
    if not chunks:
        raise ValueError("No text chunks provided for embedding.")

    logger.info(f"🔢 Generating embeddings for {len(chunks)} chunks...")
    embeddings = get_embeddings(chunks)
    logger.info(f"✅ Generated {len(embeddings)} embeddings successfully.")
    return embeddings


def process_document(text: str) -> Dict:
    """
    Full document processing pipeline:
      1. Chunk text
      2. Embed chunks

    Args:
        text (str): Extracted text from the uploaded document.
    Returns:
        Dict: Contains chunks and their embeddings.
    """
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    result = {
        "total_chunks": len(chunks),
        "chunks": chunks,
        "embeddings": embeddings
    }

    logger.info(f"📄 Document processed: {len(chunks)} chunks embedded.")
    return result

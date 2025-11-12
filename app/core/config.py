"""
core/config.py

Central configuration for the RAG-based Document QA System.
Loads environment variables, defines constants, and provides
a single source of truth for all configurable parameters.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env (if present)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

# -------------------------------
# Embedding Config (OpenAI)
# -------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing. Please set it in your .env file.")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# -------------------------------
# LLM Config (Ollama)
# -------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # Default model pulled via `ollama pull llama3`

# -------------------------------
# FAISS / Storage Config
# -------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data" / "index"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "metadata.json"


# -------------------------------
# Retrieval / Chunking Config
# -------------------------------
TOP_K = int(os.getenv("TOP_K", 4))              # Number of chunks to retrieve per query
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000)) # Default chunk size for text splitting
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))


# -------------------------------
# Logging Config
# -------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# -------------------------------
# Utility: Display current config (for debugging)
# -------------------------------
def print_config_summary():
    """Print key configuration values for quick verification."""
    print(f"\n🚀 Loaded configuration for {APP_NAME}")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Debug mode: {DEBUG}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Ollama model: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
    print(f"FAISS directory: {DATA_DIR}")
    print(f"Chunk size / overlap: {CHUNK_SIZE}/{CHUNK_OVERLAP}")
    print(f"Retrieval Top-K: {TOP_K}\n")
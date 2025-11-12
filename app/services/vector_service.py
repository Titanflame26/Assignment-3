"""
services/vector_service.py

Handles vector storage and retrieval using FAISS.
Ensures embedding dimensions match OpenAI/Ollama embeddings,
and maintains JSON-safe metadata for each stored chunk.
"""

import os
import json
import logging
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict
from app.core.config import DATA_DIR, FAISS_INDEX_PATH, METADATA_PATH
from app.services.embedding_service import get_single_embedding

logger = logging.getLogger(__name__)


class VectorService:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.index = None
        self.metadata: Dict[str, Dict] = {}
        self.dim = None
        self._load()

    # ----------------------------
    # Initialization & Persistence
    # ----------------------------
    def _load(self):
        """Load FAISS index and metadata if they exist."""
        try:
            if METADATA_PATH.exists():
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    self.dim = self.metadata.get("_dim")

            if FAISS_INDEX_PATH.exists() and self.dim:
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                logger.info(
                    f"✅ Loaded FAISS index with dim={self.dim}, total vectors={self.index.ntotal}"
                )
            else:
                logger.info("ℹ️ No existing FAISS index found. Will create a new one on first insert.")
        except Exception as e:
            logger.error(f"❌ Failed to load FAISS index or metadata: {e}")
            self.metadata = {}
            self.index = None
            self.dim = None

    def _persist_metadata(self):
        """Persist metadata dictionary to JSON file (JSON-safe)."""
        meta_copy = {}
        for k, v in self.metadata.items():
            if isinstance(v, dict):
                safe_meta = {}
                for key, val in v.items():
                    if isinstance(val, (np.integer, np.floating)):
                        val = val.item()
                    elif isinstance(val, (np.ndarray, list)):
                        continue  # skip raw embeddings
                    safe_meta[key] = val
                meta_copy[k] = safe_meta
        if self.dim:
            meta_copy["_dim"] = self.dim
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(meta_copy, f, ensure_ascii=False, indent=2)

    def _save_index(self):
        """Save FAISS index to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(FAISS_INDEX_PATH))

    # ----------------------------
    # Core Operations
    # ----------------------------
    def _ensure_index(self, dim: int):
        """Create a new FAISS index if not already initialized."""
        if self.index is None:
            logger.info(f"🧠 Initializing new FAISS index (dim={dim})")
            self.index = faiss.IndexFlatL2(dim)
            self.dim = dim

    def add_embeddings(self, embeddings: List[List[float]], metadatas: List[Dict]) -> List[int]:
        """Add new embeddings to FAISS index with metadata."""
        if not embeddings:
            raise ValueError("No embeddings to add to FAISS index.")

        embs = np.array(embeddings, dtype="float32")
        dim = embs.shape[1]
        self._ensure_index(dim)

        if self.dim != dim:
            raise ValueError(f"❌ Embedding dimension mismatch: FAISS({self.dim}) vs embeddings({dim})")

        start_id = int(self.index.ntotal)
        self.index.add(embs)

        # Store metadata for each vector
        for i, meta in enumerate(metadatas):
            vid = str(start_id + i)
            meta = dict(meta)
            meta["embedding_dim"] = dim
            self.metadata[vid] = meta

        self._save_index()
        self._persist_metadata()

        logger.info(f"✅ Added {len(embeddings)} vectors. Total vectors now: {self.index.ntotal}")
        return list(range(start_id, start_id + len(embeddings)))

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        """Perform similarity search for a given text query."""
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("FAISS index is empty. Upload documents first.")

        query_emb = np.array([get_single_embedding(query)], dtype="float32")
        if query_emb.shape[1] != self.dim:
            raise ValueError(f"❌ Query embedding dimension mismatch: {query_emb.shape[1]} vs index {self.dim}")

        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            meta = self.metadata.get(str(idx))
            if not meta:
                continue
            results.append(
                {
                    "vector_id": int(idx),
                    "distance": float(dist),
                    "text": meta.get("text", ""),
                    "doc_id": meta.get("doc_id"),
                    "source": meta.get("source"),
                    "chunk_id": meta.get("chunk_id"),
                }
            )
        logger.info(f"🔍 Retrieved {len(results)} results for query.")
        return results

    def list_documents(self) -> List[Dict]:
        """Return JSON-safe summary of all indexed documents."""
        docs: Dict[str, Dict] = {}
        for meta in self.metadata.values():
            if not isinstance(meta, dict):
                continue
            doc_id = str(meta.get("doc_id")) if meta.get("doc_id") else None
            if not doc_id:
                continue
            src = str(meta.get("source", "unknown"))
            docs.setdefault(doc_id, {"doc_id": doc_id, "source": src, "chunks": 0})
            docs[doc_id]["chunks"] += 1

        result = list(docs.values())
        logger.info(f"📚 list_documents -> {len(result)} docs found")
        return result

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and rebuild FAISS index without its embeddings."""
        remaining = []
        for meta in self.metadata.values():
            if not isinstance(meta, dict):
                continue
            if meta.get("doc_id") != doc_id:
                remaining.append(meta)

        if len(remaining) == len(self.metadata):
            logger.warning(f"⚠️ No document found with id={doc_id}")
            return False

        if not remaining:
            # Reset everything
            self.index = None
            self.metadata = {}
            self.dim = None
            if FAISS_INDEX_PATH.exists():
                FAISS_INDEX_PATH.unlink()
            if METADATA_PATH.exists():
                METADATA_PATH.unlink()
            logger.info("🧹 All documents removed. FAISS index reset.")
            return True

        # Rebuild FAISS
        embeddings = [np.array(m["embedding"], dtype="float32") for m in remaining if "embedding" in m]
        if not embeddings:
            logger.error("❌ No valid embeddings found during rebuild.")
            return False

        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.vstack(embeddings))

        new_meta = {}
        for i, meta in enumerate(remaining):
            new_meta[str(i)] = meta

        self.metadata = new_meta
        self.dim = dim
        self._save_index()
        self._persist_metadata()

        logger.info(f"🗑️ Deleted doc_id={doc_id}. Rebuilt index with {self.index.ntotal} vectors.")
        return True

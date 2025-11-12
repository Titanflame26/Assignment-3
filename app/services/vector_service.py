"""
services/vector_service.py

Handles vector storage and retrieval using FAISS.
Ensures embedding dimensions match OpenAI embeddings,
and maintains metadata for each stored chunk.
"""

import os
import json
import logging
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from app.core.config import DATA_DIR, FAISS_INDEX_PATH, METADATA_PATH
from app.services.embedding_service import get_single_embedding

logger = logging.getLogger(__name__)


class VectorService:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.index = None
        self.metadata: Dict[str, Dict] = {}
        self.dim = None  # embedding dimension
        self._load()

    # ----------------------------
    # Initialization & Persistence
    # ----------------------------
    def _load(self):
        """Load FAISS index and metadata if they exist."""
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                self.dim = self.metadata.get("_dim")

        if FAISS_INDEX_PATH.exists() and self.dim:
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            logger.info(f"✅ Loaded FAISS index with dim={self.dim}, total vectors={self.index.ntotal}")
        else:
            logger.info("ℹ️ No existing FAISS index found. A new one will be created on first insert.")

    def _persist_metadata(self):
        """Persist metadata dictionary to JSON file."""
        meta_copy = dict(self.metadata)
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

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict]
    ) -> List[int]:
        """
        Add new embeddings to FAISS index with metadata.

        Args:
            embeddings (List[List[float]]): List of embeddings.
            metadatas (List[Dict]): List of associated metadata.
        Returns:
            List[int]: List of assigned vector IDs.
        """
        if not embeddings:
            raise ValueError("No embeddings to add to FAISS index.")

        embs = np.array(embeddings, dtype="float32")
        dim = embs.shape[1]

        self._ensure_index(dim)

        if dim != self.dim:
            raise ValueError(f"❌ Embedding dimension mismatch: FAISS({self.dim}) vs OpenAI({dim})")

        start_id = int(self.index.ntotal)
        self.index.add(embs)

        # Add metadata for each vector
        for i, meta in enumerate(metadatas):
            vid = str(start_id + i)
            meta["embedding_dim"] = dim
            self.metadata[vid] = meta

        self._save_index()
        self._persist_metadata()

        logger.info(f"✅ Added {len(embeddings)} vectors to FAISS index. Total = {self.index.ntotal}")
        return list(range(start_id, start_id + len(embeddings)))

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        """
        Perform similarity search using the given query text.

        Args:
            query (str): Search query text.
            top_k (int): Number of results to return.
        Returns:
            List[Dict]: Top matching chunks with metadata and distance.
        """
        if self.index is None:
            raise ValueError("FAISS index is empty. No documents available for search.")

        query_emb = np.array([get_single_embedding(query)], dtype="float32")

        if query_emb.shape[1] != self.dim:
            raise ValueError(f"❌ Query embedding dimension mismatch: {query_emb.shape[1]} vs index {self.dim}")

        distances, indices = self.index.search(query_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if str(idx) in self.metadata:
                meta = self.metadata[str(idx)]
                results.append({
                    "vector_id": idx,
                    "distance": float(dist),
                    "text": meta.get("text", ""),
                    "doc_id": meta.get("doc_id"),
                    "source": meta.get("source"),
                    "chunk_id": meta.get("chunk_id")
                })

        logger.info(f"🔍 Retrieved {len(results)} results for query.")
        return results

    def list_documents(self) -> List[Dict]:
        """List all unique documents stored in the index."""
        docs = {}
        for meta in self.metadata.values():
            doc_id = meta.get("doc_id")
            if not doc_id:
                continue
            docs.setdefault(doc_id, {"doc_id": doc_id, "source": meta.get("source"), "chunks": 0})
            docs[doc_id]["chunks"] += 1

        return list(docs.values())

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and rebuild FAISS index without its embeddings.
        """
        remaining = []
        new_meta = {}
        for vid, meta in self.metadata.items():
            if meta.get("doc_id") != doc_id and not vid.startswith("_"):
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

        # Rebuild index
        all_embeddings = [meta["embedding"] for meta in remaining]
        dim = len(all_embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        arr = np.array(all_embeddings, dtype="float32")
        self.index.add(arr)

        for i, meta in enumerate(remaining):
            new_meta[str(i)] = meta

        self.metadata = new_meta
        self.dim = dim
        self._save_index()
        self._persist_metadata()

        logger.info(f"🗑️ Deleted document {doc_id}. Rebuilt index with {self.index.ntotal} vectors remaining.")
        return True

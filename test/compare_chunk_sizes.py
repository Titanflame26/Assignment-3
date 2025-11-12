"""
experiments/compare_chunk_sizes.py

Test and compare retrieval quality for different chunk sizes and overlaps.
This script:
  1. Reads a single document.
  2. Splits it using multiple (chunk_size, chunk_overlap) settings.
  3. Embeds each version using OpenAI embeddings.
  4. Stores them in temporary FAISS indexes.
  5. Runs a sample query against each index.
"""

import os
import tempfile
from pathlib import Path
import numpy as np
from app.services.file_readers import read_file
from app.services.docs_service import chunk_text, embed_chunks
from app.services.vector_service import VectorService
from app.core.config import OPENAI_API_KEY
from app.services.embedding_service import get_single_embedding

def compare_chunk_configs(
    file_path: str,
    query: str,
    chunk_sizes: list[int] = [500, 1000, 1500, 2000],
    chunk_overlap: int = 200,
    top_k: int = 3
):
    """
    Runs retrieval comparison for multiple chunk size configurations.

    Args:
        file_path (str): Path to input document (.txt or .pdf)
        query (str): Query text to test retrieval
        chunk_sizes (list[int]): List of chunk sizes to evaluate
        chunk_overlap (int): Overlap size for all configurations
        top_k (int): Number of retrieved results per query
    """
    print(f"\n🔍 Comparing chunk configurations for: {Path(file_path).name}")
    print(f"Query: {query}\n{'-' * 80}")

    # Read document once
    text = read_file(file_path)

    for size in chunk_sizes:
        print(f"\n🧩 Testing chunk_size={size}, overlap={chunk_overlap}...")

        # Create temporary folder for each experiment
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["DATA_DIR"] = tmpdir  # isolate each FAISS index

            # Step 1: Chunk text
            chunks = chunk_text(text)
            print(f"  • Generated {len(chunks)} chunks")

            # Step 2: Embed chunks
            embeddings = embed_chunks(chunks)
            dim = len(embeddings[0])
            print(f"  • Embedding dimension: {dim}")

            # Step 3: Initialize FAISS
            vs = VectorService()

            metadatas = [
                {
                    "doc_id": f"exp_{size}",
                    "source": Path(file_path).name,
                    "chunk_id": i,
                    "text": chunk,
                    "embedding": emb
                }
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
            ]
            vs.add_embeddings(embeddings, metadatas)

            # Step 4: Test retrieval
            results = vs.search(query, top_k=top_k)
            print(f"  • Retrieved {len(results)} chunks")

            # Step 5: Print top results
            for r in results:
                snippet = r["text"][:100].replace("\n", " ")
                print(f"    ↳ [Dist={r['distance']:.4f}] {snippet}...")

            # Calculate average distance (lower = better)
            if results:
                avg_dist = np.mean([r["distance"] for r in results])
                print(f"  • Avg distance: {avg_dist:.4f}")
            else:
                print("  ⚠️ No retrieval results for this configuration.")

    print("\n✅ Comparison complete.")


if __name__ == "__main__":
    # Example usage
    test_file = "D:/Projects to do/Obulaneni_Bharadwaj_DataScience/My Resume.pdf"
    test_query = "What is the educational background?"
    compare_chunk_configs(
        file_path=test_file,
        query=test_query,
        chunk_sizes=[500, 1000, 1500],
        chunk_overlap=200,
        top_k=3
    )

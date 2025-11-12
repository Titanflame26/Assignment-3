"""
api/upload.py
Handles document uploads:
1. Extract text
2. Chunk + embed (OpenAI/Ollama fallback)
3. Store embeddings & metadata in FAISS
4. Return summary response
"""

import os
import tempfile
import logging
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.file_readers import read_file
from app.services.docs_service import process_document
from app.services.vector_service import VectorService
from app.models.schemas import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize FAISS vector service
vector_service = VectorService()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document, extract text, generate embeddings, and store them in FAISS.
    """
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    allowed_exts = [".pdf", ".txt"]

    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, filename)
            with open(temp_path, "wb") as f:
                contents = await file.read()
                f.write(contents)

            # Step 1: Extract text
            text = read_file(temp_path)
            if not text.strip():
                raise HTTPException(status_code=400, detail="No readable text found in the document.")

            # Step 2: Process (chunk + embed)
            result = process_document(text)
            chunks = result["chunks"]
            embeddings = result["embeddings"]

            # Step 3: Prepare metadata
            doc_id = str(uuid.uuid4())[:8]  # short unique ID
            metadatas = [
                {
                    "doc_id": doc_id,
                    "source": filename,
                    "chunk_id": i,
                    "text": chunk,
                    "embedding": emb
                }
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
            ]

            # Step 4: Store in FAISS
            vector_service.add_embeddings(embeddings, metadatas)
            logger.info(f"✅ Stored document '{filename}' in FAISS with {len(chunks)} chunks.")

            # Step 5: Return response
            return UploadResponse(
                filename=filename,
                extension=ext,
                total_chunks=len(chunks),
                embedding_dimension=len(embeddings[0]) if embeddings else 0,
                message="File processed, embedded, and stored successfully."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing uploaded file.")
        raise HTTPException(status_code=500, detail=f"File processing failed: {e}")

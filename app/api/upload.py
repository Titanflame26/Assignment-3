"""
api/upload.py
Handles document uploads:
1. Extract text
2. Chunk + embed (OpenAI/Ollama fallback)
3. Return summary response
"""

import os
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.file_readers import read_file
from services.docs_service import process_document
from models.schemas import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
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

            text = read_file(temp_path)
            if not text.strip():
                raise HTTPException(status_code=400, detail="No readable text found.")

            result = process_document(text)

            return UploadResponse(
                filename=filename,
                extension=ext,
                total_chunks=result["total_chunks"],
                embedding_dimension=len(result["embeddings"][0]) if result["embeddings"] else 0,
                message="File processed successfully and embeddings generated."
            )

    except Exception as e:
        logger.exception("Error processing uploaded file.")
        raise HTTPException(status_code=500, detail=f"File processing failed: {e}")

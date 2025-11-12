"""
api/upload.py

Handles document uploads:
1. Accepts PDF or TXT file.
2. Extracts text.
3. Splits into chunks & generates embeddings.
"""

import os
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.file_readers import read_file
from services.docs_service import process_document

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document, extract text, chunk it, and generate embeddings.
    Returns metadata about the processed document.
    """
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    allowed_exts = [".pdf", ".txt"]

    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, filename)

            # Save uploaded file locally
            with open(temp_path, "wb") as f:
                contents = await file.read()
                f.write(contents)

            # Extract text
            text = read_file(temp_path)
            if not text.strip():
                raise HTTPException(status_code=400, detail="No readable text found in the document.")

            # Process document (chunk + embed)
            result = process_document(text)

            logger.info(
                f"✅ File processed: {filename} | "
                f"Chunks: {result['total_chunks']} | "
                f"Embedding Dim: {len(result['embeddings'][0]) if result['embeddings'] else 'N/A'}"
            )

            return {
                "filename": filename,
                "extension": ext,
                "total_chunks": result["total_chunks"],
                "embedding_dimension": len(result["embeddings"][0]) if result["embeddings"] else 0,
                "message": "File processed successfully. Chunks embedded.",
            }

    except ValueError as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error while processing file")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

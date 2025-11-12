"""
api/documents.py
List and delete indexed documents.
"""

import logging
from fastapi import APIRouter, HTTPException
from app.services.vector_service import VectorService
from app.models.schemas import DocumentListResponse, DocumentDeleteResponse

logger = logging.getLogger(__name__)
router = APIRouter()

vector_service = VectorService()

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    try:
        docs = vector_service.list_documents()
        return DocumentListResponse(documents=docs)
    except Exception as e:
        logger.exception("Error listing documents.")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {e}")

@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str):
    try:
        success = vector_service.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Document with id '{doc_id}' not found.")
        return DocumentDeleteResponse(message=f"Document {doc_id} deleted successfully.")
    except Exception as e:
        logger.exception("Error deleting document.")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")

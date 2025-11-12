"""
api/documents.py

Endpoints for managing indexed documents:
1. List all indexed documents and their metadata.
2. Delete a document from FAISS and rebuild the index.
"""

import logging
from fastapi import APIRouter, HTTPException
from services.vector_service import VectorService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize FAISS vector service
vector_service = VectorService()


@router.get("/documents")
async def list_documents():
    """
    List all documents currently indexed in FAISS.

    Returns:
        List[Dict]: List of documents with doc_id, source, and chunk count.
    """
    try:
        docs = vector_service.list_documents()
        if not docs:
            logger.info("ℹ️ No documents indexed yet.")
            return {"message": "No documents found in the index.", "documents": []}

        logger.info(f"📚 Retrieved {len(docs)} indexed documents.")
        return {"documents": docs}

    except Exception as e:
        logger.exception("❌ Failed to list documents.")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {e}")


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a specific document and rebuild FAISS index.

    Args:
        doc_id (str): Unique ID of the document to delete.
    Returns:
        dict: Confirmation message.
    """
    try:
        logger.info(f"🗑️ Received request to delete document: {doc_id}")
        success = vector_service.delete_document(doc_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Document with id '{doc_id}' not found.")

        logger.info(f"✅ Deleted document {doc_id} successfully.")
        return {"message": f"Document {doc_id} deleted successfully."}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("❌ Error deleting document from FAISS index.")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")

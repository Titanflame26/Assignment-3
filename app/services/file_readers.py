"""
services/file_readers.py

Handles reading and extracting text from different document types.
Currently supports: .txt and .pdf
"""

import os
from PyPDF2 import PdfReader

def read_file(file_path: str) -> str:
    """
    Detects file type and extracts text accordingly.
    Supports .txt and .pdf formats.

    Args:
        file_path (str): Path to the uploaded file.
    Returns:
        str: Extracted plain text from the document.
    Raises:
        ValueError: If file type is unsupported or no text could be extracted.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return read_txt(file_path)
    elif ext == ".pdf":
        return read_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only .txt and .pdf are supported.")


def read_txt(file_path: str) -> str:
    """Reads and returns text content from a .txt file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading text file: {e}")


def read_pdf(file_path: str) -> str:
    """Extracts and returns text from all pages of a PDF."""
    try:
        reader = PdfReader(file_path)
        text_pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(text_pages).strip()
        if not text:
            raise ValueError("No text extracted from PDF (possibly scanned or image-based).")
        return text
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {e}")

import os
from PyPDF2 import PdfReader
from typing import List

def load_documents(file_paths: List[str]) -> List[str]:
    """
    Load and extract text from a list of PDF documents.

    Args:
        file_paths (List[str]): List of file paths to the PDF documents.

    Returns:
        List[str]: List of text content from each document.
    """
    texts = []
    for path in file_paths:
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            texts.append(text)
        except Exception as e:
            print(f"Error loading document {path}: {e}")
    return texts

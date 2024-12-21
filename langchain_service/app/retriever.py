from typing import List

def split_text_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """
    Split a large text into smaller chunks.

    Args:
        text (str): The input text.
        chunk_size (int): The maximum size of each chunk.

    Returns:
        List[str]: List of text chunks.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def retrieve_relevant_chunks(chunks: List[str], query: str) -> List[str]:
    """
    Retrieve the most relevant text chunks based on a query.

    Args:
        chunks (List[str]): List of text chunks.
        query (str): The search query.

    Returns:
        List[str]: List of relevant chunks.
    """
    # For simplicity, return all chunks; replace with semantic search later
    return chunks

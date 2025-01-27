from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

def retrieve_relevant_chunks(document_store, query: str, top_k: int = 5) -> List[str]:
    """
    Retrieve the most relevant text chunks using pre-computed embeddings.

    Args:
        document_store: DocumentStore instance containing pre-computed embeddings
        query (str): The search query
        top_k (int): Number of most relevant chunks to return

    Returns:
        List[str]: List of top-k most relevant chunks
    """
    query_embedding = document_store.get_query_embedding(query)
    chunk_embeddings = document_store.get_embeddings()
    chunks = document_store.get_chunks()
    
    # Calculate cosine similarity between query and all chunks
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # Get indices of top-k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return the most relevant chunks
    return [chunks[i] for i in top_indices]
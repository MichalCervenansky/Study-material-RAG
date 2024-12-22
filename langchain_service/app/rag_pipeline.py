from typing import List
from app.document_loader import load_documents
from app.retriever import split_text_into_chunks, retrieve_relevant_chunks
from app.ollama_integration import query_ollama

def rag_pipeline(file_paths: List[str], query: str) -> str:
    """
    Process documents and answer a query using the RAG pipeline.

    Args:
        file_paths (List[str]): List of document file paths.
        query (str): The user query.

    Returns:
        str: The generated answer.
    """
    documents = []
    all_chunks = [chunk for doc in documents for chunk in split_text_into_chunks(doc)]
    relevant_chunks = retrieve_relevant_chunks(all_chunks, query)
    combined_context = " ".join(relevant_chunks)
    prompt = f"Context: {combined_context}\n\nQuestion: {query}\n\nAnswer:"
    return query_ollama(prompt)

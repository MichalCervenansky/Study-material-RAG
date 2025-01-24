import os
from typing import List, Iterator
from app.ollama_integration import OllamaAPI
from app.document_loader import load_documents
from app.retriever import split_text_into_chunks, retrieve_relevant_chunks
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def rag_pipeline(file_paths: List[str], query: str) -> Iterator[str]:
    """
    Process documents and answer a query using the RAG pipeline.

    Args:
        file_paths (List[str]): List of document file paths.
        query (str): The user query.

    Returns:
        str: The generated answer.
    """
    # Load documents from file paths
    documents = load_documents(file_paths)

    # Create chunks from all documents
    all_chunks = []
    for doc in documents:
        chunks = split_text_into_chunks(doc)
        all_chunks.extend(chunks)

    # Retrieve relevant chunks based on the query
    relevant_chunks = retrieve_relevant_chunks(all_chunks, query)

    # Combine chunks into context
    combined_context = " ".join(relevant_chunks)

    # Create prompt and generate response
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {combined_context}"},
        {"role": "user", "content": f"Question: {query}"},
        {"role": "assistant", "content": "Answer:"}
    ]

    # Get the model from the environment variable
    model = os.getenv("OLLAMA_MODEL")

    # Initialize OllamaAPI and generate response
    ollama_api = OllamaAPI()
    return ollama_api.chat(prompt, model=model)

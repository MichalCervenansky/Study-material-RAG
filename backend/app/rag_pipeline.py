import os
from typing import List, Iterator
from app.ollama_integration import OllamaAPI
from app.retriever import retrieve_relevant_chunks
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def rag_pipeline(document_store, query: str) -> Iterator[str]:
    """
    Answer a query using the RAG pipeline with pre-computed embeddings.

    Args:
        document_store: DocumentStore instance containing pre-computed embeddings
        query (str): The user query.

    Returns:
        Iterator[str]: The generated answer chunks.
    """
    # Retrieve relevant chunks based on the query
    relevant_chunks = retrieve_relevant_chunks(document_store, query)

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

import os
from typing import Iterator
from app.ollama_integration import OllamaAPI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def rag_pipeline(document_store, query: str) -> Iterator[str]:
    """
    RAG pipeline using ChromaDB for retrieval.
    """
    # Query ChromaDB for relevant chunks
    results = document_store.query_documents(query)
    
    if results and results['documents']:
        # Get the first list of documents from results
        relevant_chunks = results['documents'][0]
        combined_context = " ".join(relevant_chunks)

        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {combined_context}"},
            {"role": "user", "content": f"Question: {query}"},
            {"role": "assistant", "content": "Answer:"}
        ]

        model = os.getenv("OLLAMA_MODEL")
        ollama_api = OllamaAPI()
        return ollama_api.chat(prompt, model=model)
    
    return iter(["No relevant documents found."])

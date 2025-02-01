import os
from typing import AsyncGenerator
from app.ollama_integration import OllamaAPI
from dotenv import load_dotenv

load_dotenv()


async def rag_pipeline(document_store, query: str) -> AsyncGenerator[str, None]:
    """
    Async RAG pipeline with proper streaming
    """
    results = document_store.query_documents(query)

    if not results or not results['documents']:
        yield "No relevant documents found."
        return

    relevant_chunks = results['documents'][0]
    combined_context = " ".join(relevant_chunks)

    prompt = [
        {"role": "system",
         "content": "You are a helpful assistant. Answer step by step based on the context provided."},
        {"role": "user", "content": f"Context: {combined_context}\n\nQuestion: {query}"}
    ]

    model = os.getenv("OLLAMA_MODEL", "")
    ollama_api = OllamaAPI()

    async for token in ollama_api.chat(prompt, model=model):
        yield token

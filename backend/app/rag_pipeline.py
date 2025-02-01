import os
from typing import AsyncGenerator, List
from app.ollama_integration import OllamaAPI
from dotenv import load_dotenv

load_dotenv()


async def rag_pipeline(document_store, query: str, messages: List[dict] = None, previous_chunks: List[str] = None) -> AsyncGenerator[str, None]:
    """
    Async RAG pipeline with proper streaming
    """
    # Get new relevant chunks
    results = document_store.query_documents(query)
    current_chunks = results['documents'][0] if results['documents'] else []
    
    # Combine with previous context if available
    all_chunks = current_chunks
    if previous_chunks:
        all_chunks = list(set(current_chunks + previous_chunks))  # Remove duplicates
    
    combined_context = " ".join(all_chunks)

    prompt = [
        {
            "role": "system",
            "content": f"""**RAG Assistant Guidelines**
       1. Analyze the context thoroughly before answering
       2. Use ONLY verified information from provided documents
       3. If information is missing, state "This is not covered in my documentation"
       4. Format response with:
          - Clear headings using ##
          - Bullet points for lists
          - Code blocks where applicable
          - Citations like [Document X] after claims

       **Processing Steps**
       1. Identify key entities and relationships
       2. Cross-reference multiple document sections
       3. Verify temporal consistency
       4. Check for contradictory information

       ### CONTEXT ###
       {combined_context}

       ### QUESTION ###
       {query}"""
        },
        {
            "role": "user",
            "content": "Please provide a comprehensive answer with document citations."
        }
    ]

    model = os.getenv("OLLAMA_MODEL", "")
    ollama_api = OllamaAPI()

    async for token in ollama_api.chat(prompt, model=model):
        yield token

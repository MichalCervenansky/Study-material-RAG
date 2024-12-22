from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import rag_pipeline

import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_service(request: QueryRequest):
    """
    Answer a question using the RAG pipeline.
    """
    # Replace with actual file paths in production
    file_paths = ["/path/to/processed/documents"]
    answer = rag_pipeline(file_paths, request.question)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
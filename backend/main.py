import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.rag_pipeline import rag_pipeline
from app.document_store import DocumentStore
import json
import uvicorn

# Initialize document store globally
document_store = DocumentStore()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load documents
    document_folder = "/Users/I752271/PycharmProjects/Study-material-RAG/documents/"
    file_paths = [os.path.join(document_folder, file) 
                 for file in os.listdir(document_folder) 
                 if file.endswith(".pdf")]
    document_store.load_documents(file_paths)
    yield
    # Cleanup (if needed)
    
# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_service(request: QueryRequest):
    """
    Answer a question using the RAG pipeline.
    """
    async def generate():
        for chunk in rag_pipeline(document_store, request.question):
            yield f"data: {json.dumps({'answer': chunk})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
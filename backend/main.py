import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.rag_pipeline import rag_pipeline
from app.document_store import ChromaDocStore
from typing import List, Dict, Any
import json
import uvicorn

# Initialize ChromaDB store
chroma_store = ChromaDocStore()

# Initialize FastAPI
app = FastAPI()

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

class DocumentUploadRequest(BaseModel):
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    ids: List[str]


@app.post("/query")
async def query_service(request: QueryRequest):
    """
    Streaming endpoint with proper async handling
    """

    async def generate():
        try:
            # Start streaming immediately
            async for chunk in rag_pipeline(chroma_store, request.question):
                if chunk:
                    message = json.dumps({"answer": chunk})
                    yield f"data: {message}\n\n"

        except Exception as e:
            error_msg = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_msg}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/documents/add")
async def add_documents(request: DocumentUploadRequest):
    success = chroma_store.add_documents(
        request.documents,
        request.metadatas,
        request.ids
    )
    if success:
        return {"status": "success", "message": f"Added {len(request.documents)} documents"}
    return {"status": "error", "message": "Failed to add documents"}

@app.get("/documents")
async def get_documents():
    results = chroma_store.get_all_documents()
    return results

@app.post("/documents/clear")
async def clear_documents():
    try:
        success = chroma_store.clear_documents()
        if success:
            return {"status": "success", "message": "Documents cleared successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to clear documents: {str(e)}"}
    return {"status": "error", "message": "Failed to clear documents"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
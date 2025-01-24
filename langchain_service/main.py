from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.rag_pipeline import rag_pipeline
import json

import uvicorn

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

@app.post("/query")
async def query_service(request: QueryRequest):
    """
    Answer a question using the RAG pipeline.
    """
    async def generate():
        file_paths = ["/Users/I752271/PycharmProjects/Study-material-RAG/documents/CV_Michal_Cervenansky.pdf"]
        for chunk in rag_pipeline(file_paths, request.question):
            yield f"data: {json.dumps({'answer': chunk})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
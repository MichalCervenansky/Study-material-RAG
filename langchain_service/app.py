from fastapi import FastAPI, UploadFile, Form
from app.rag_pipeline import rag_pipeline

app = FastAPI()

@app.post("/upload")
async def upload_files(files: List[UploadFile]):
    """
    Upload PDF files to process.
    """
    file_paths = []
    for file in files:
        path = f"/tmp/{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())
        file_paths.append(path)
    return {"message": f"Uploaded {len(file_paths)} files successfully"}

@app.post("/query")
async def query_service(question: str = Form(...)):
    """
    Answer a question using the RAG pipeline.
    """
    # Replace with actual file paths in production
    file_paths = ["/path/to/processed/documents"]
    answer = rag_pipeline(file_paths, question)
    return {"answer": answer}

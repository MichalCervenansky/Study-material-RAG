# Retrieval-Augmented Generation (RAG) Service

A RAG service using LangChain, Ollama LLM, and Streamlit for processing and querying study materials.

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd Study-material-RAG
```

2. Configure environment:
Copy the example environment file and modify as needed:
```bash
cp .env.example .env
```

3. Start the services:
```bash
docker-compose up -d
```

This will start:
- Backend service at http://localhost:8000
- Frontend interface at http://localhost:8501

## Prerequisites

1. Install Ollama from https://ollama.ai
2. Pull the required model:
```bash
ollama pull deepseek-r1:latest
```

## Development Setup

### Running Services Locally

1. Start Ollama:
```bash
ollama run deepseek-r1:latest
```

2. Start the backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

3. Start the frontend:
```bash
cd streamlit_frontend
pip install -r requirements.txt
streamlit run Home.py
```

## Project Structure
```
./
├── backend/              # FastAPI + LangChain service
├── streamlit_frontend/   # Streamlit UI
├── docker-compose.yml   # Container orchestration
└── .env                 # Environment configuration
```

## Usage
1. Access the Streamlit interface at http://localhost:8501
2. Upload PDF documents in the Database section
3. Ask questions about your documents in the Chat section

## Environment Variables
Key configurations in `.env`:
- `BACKEND_URL`: Backend service URL
- `OLLAMA_BASE_URL`: Ollama service URL
- `OLLAMA_MODEL`: LLM model to use
- `CHUNK_SIZE`: Document chunking size
- `CHUNK_OVERLAP`: Overlap between chunks

## Features
- **Document Upload**: Upload PDF study materials.
- **Text Retrieval**: Extract relevant sections of the documents.
- **Question Answering**: Generate answers to user queries using RAG and Ollama LLM.
- **User Interface**: A Streamlit-based frontend for easy interaction.

## API Endpoints
The LangChain service exposes the following endpoints:
- **`POST /upload`**: Upload PDF files for processing.
- **`POST /query`**: Send a question to the RAG pipeline and get an answer.

## Testing
Unit tests for the backend can be run using:
```bash
pytest backend/tests
```

## Deployment
Deploy the services using Docker Compose or container orchestration platforms like Kubernetes. Use the provided `Dockerfile` and `docker-compose.yml` for deployment.

## Acknowledgments
- [LangChain](https://github.com/hwchase17/langchain)
- [Ollama](https://ollama.ai/)
- [Streamlit](https://streamlit.io/)

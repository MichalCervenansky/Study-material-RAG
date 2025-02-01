# Streamlit Frontend for RAG Service

This is the frontend interface for the Retrieval-Augmented Generation (RAG) service. It allows users to upload PDFs and ask questions interactively.

## Features
- Upload multiple PDF documents or ZIP files containing PDFs
- Interactive chat interface with context-aware responses
- Document database management
- Streaming responses from the LLM

## Prerequisites
- Python 3.10 or higher
- Backend service running (default: localhost:8000)
- Ollama service running (default: localhost:11434)

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   Create a `.env` file with:
   ```properties
   BACKEND_URL=http://localhost:8000
   ```

3. Run the application:
   ```bash
   streamlit run Home.py
   ```

## Docker Deployment
```bash
docker build -t rag-frontend .
docker run -p 8501:8501 rag-frontend
```

## Usage
1. Access the application at http://localhost:8501
2. Use the Database page to upload and manage documents
3. Navigate to the Chat page to ask questions about your documents

## Pages
- Home: Welcome page and getting started guide
- Database: Document management interface
- Chat: Interactive QA interface

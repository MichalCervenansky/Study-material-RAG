# LangChain RAG Service

This service implements a Retrieval-Augmented Generation (RAG) system using FastAPI, LangChain, and Ollama.

## Features
- Document management with ChromaDB vector store
- Streaming responses using Server-Sent Events (SSE)
- Async RAG pipeline with Ollama LLM integration
- Configurable document chunking
- CORS-enabled API endpoints

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

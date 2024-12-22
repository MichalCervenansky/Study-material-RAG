
# Retrieval-Augmented Generation (RAG) Service

This project implements a Retrieval-Augmented Generation (RAG) service using LangChain, Ollama LLM, and Streamlit. It allows users to upload study materials as PDFs, ask questions, and receive insightful answers.

## Project Structure
```
rag-service/
├── langchain_service/       # Backend service for document processing and RAG pipeline
│   ├── app/                 # Core application logic
│   ├── tests/               # Unit tests for backend
│   ├── Dockerfile           # Docker setup for LangChain service
│   ├── requirements.txt     # Python dependencies for LangChain service
│   └── main.py               # FastAPI entry point
├── streamlit_frontend/      # Streamlit-based user interface
│   ├── pages/               # Optional additional pages for Streamlit
│   ├── Dockerfile           # Docker setup for Streamlit frontend
│   ├── requirements.txt     # Python dependencies for Streamlit
│   └── main.py              # Main Streamlit application entry point
├── docker-compose.yml       # Orchestrates the backend and frontend
├── .gitignore               # Files and directories to ignore in version control
└── README.md                # Project overview and instructions
```

## Features
- **Document Upload**: Upload PDF study materials.
- **Text Retrieval**: Extract relevant sections of the documents.
- **Question Answering**: Generate answers to user queries using RAG and Ollama LLM.
- **User Interface**: A Streamlit-based frontend for easy interaction.

## Setup and Installation

### Prerequisites
- Docker and Docker Compose installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/rag-service.git
cd rag-service
```

### Step 2: Build and Start Services
Use Docker Compose to build and run the backend and frontend services:
```bash
docker-compose up --build
```

This will start:
- The LangChain service at `http://localhost:8000`.
- The Streamlit frontend at `http://localhost:8501`.

### Step 3: Interact with the Application
1. Open your browser and go to `http://localhost:8501`.
2. Upload PDF files and ask questions.

## API Endpoints
The LangChain service exposes the following endpoints:
- **`POST /upload`**: Upload PDF files for processing.
- **`POST /query`**: Send a question to the RAG pipeline and get an answer.

## Development

### Backend Development
1. Navigate to the `langchain_service/` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the service locally:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

### Frontend Development
1. Navigate to the `streamlit_frontend/` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app locally:
   ```bash
   streamlit run main.py
   ```

### Testing
Unit tests for the backend can be run using:
```bash
pytest langchain_service/tests
```

## Deployment
Deploy the services using Docker Compose or container orchestration platforms like Kubernetes. Use the provided `Dockerfile` and `docker-compose.yml` for deployment.

## Acknowledgments
- [LangChain](https://github.com/hwchase17/langchain)
- [Ollama](https://ollama.ai/)
- [Streamlit](https://streamlit.io/)

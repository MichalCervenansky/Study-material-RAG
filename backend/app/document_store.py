from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from app.document_loader import load_documents
from app.retriever import split_text_into_chunks

class DocumentStore:
    def __init__(self):
        self.chunks: List[str] = []
        self.chunk_embeddings: np.ndarray = None
        print("Loading SentenceTransformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_documents(self, file_paths: List[str]) -> None:
        """Load documents and compute embeddings."""
        # Load and chunk documents
        documents = load_documents(file_paths)
        self.chunks = []
        for doc in documents:
            self.chunks.extend(split_text_into_chunks(doc))
        
        # Compute embeddings for all chunks
        self.chunk_embeddings = self.model.encode(self.chunks)

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query."""
        return self.model.encode([query])

    def get_chunks(self) -> List[str]:
        """Get all chunks."""
        return self.chunks

    def get_embeddings(self) -> np.ndarray:
        """Get all embeddings."""
        return self.chunk_embeddings

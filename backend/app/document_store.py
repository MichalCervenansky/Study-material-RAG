import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from .logger_config import get_logger, log_time
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = get_logger(__name__)

class ChromaDocStore:
    def __init__(self):
        self.settings = Settings(
            allow_reset=os.getenv('CHROMA_ALLOW_RESET', 'true').lower() == 'true',
            anonymized_telemetry=os.getenv('CHROMA_ANONYMIZED_TELEMETRY', 'false').lower() == 'true',
            is_persistent=os.getenv('CHROMA_IS_PERSISTENT', 'true').lower() == 'true'
        )
        
        self.client = chromadb.Client(self.settings)
        self.collection_name = "documents"
        # Use default embedding function from ChromaDB
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n##", "\n\n", "\n", ". ", " ", ""]
        )
        logger.info(f"Initialized ChromaDocStore with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    @log_time(logger)
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str] = None) -> bool:
        try:
            # Validate input lengths match
            if len(documents) != len(metadatas):
                raise ValueError(f"Number of documents ({len(documents)}) must match number of metadatas ({len(metadatas)})")

            # Get current collection size for ID generation
            current_docs = self.collection.get()
            start_idx = len(current_docs['ids']) if current_docs['ids'] else 0
            
            # Generate sequential IDs
            generated_ids = [f"doc_{i}" for i in range(start_idx, start_idx + len(documents))]
            
            logger.info(f"Adding {len(documents)} documents with IDs {generated_ids[0]} to {generated_ids[-1]}")
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=generated_ids
            )
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def get_chunking_config(self):
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }

    def get_all_documents(self):
        return self.collection.get()

    @log_time(logger)
    def query_documents(self, query: str, n_results: int = 5):
        logger.info(f"Querying documents with: {query[:100]}...")
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Log retrieved chunks and their distances
        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i] if 'distances' in results else 'N/A'
            metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
            logger.info(f"Retrieved chunk {i + 1}/{n_results}:")
            logger.info(f"  Distance: {distance}")
            logger.info(f"  Metadata: {metadata}")
            logger.info(f"  Content: {results['documents'][0][i][:50]}...")
            
        return results

    @log_time(logger)
    def clear_documents(self):
        logger.info("Clearing all documents")
        try:
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
            return True
        except Exception as e:
            print(f"Error clearing documents: {e}")
            return False

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from .logger_config import get_logger, log_time
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader

# Get the project root directory (where .env is located)
root_dir = Path(__file__).resolve().parents[2]  # Go up 2 levels from document_store.py
env_path = root_dir / '.env'

# Load the environment variables from the root .env file
load_dotenv(dotenv_path=env_path)

logger = get_logger(__name__)

logger.info(f"Loading environment variables from: {env_path}")
logger.debug(f"CHUNK_SIZE: {os.getenv('CHUNK_SIZE')}")
logger.debug(f"CHUNK_OVERLAP: {os.getenv('CHUNK_OVERLAP')}")

class ChromaDocStore:
    def __init__(self):
        self.settings = Settings(
            allow_reset=os.getenv('CHROMA_ALLOW_RESET', 'true').lower() == 'true',
            anonymized_telemetry=os.getenv('CHROMA_ANONYMIZED_TELEMETRY', 'false').lower() == 'true',
            is_persistent=os.getenv('CHROMA_IS_PERSISTENT', 'true').lower() == 'true'
        )
        
        self.client = chromadb.Client(self.settings)
        self.collection_name = "documents"
        
        # Load configuration from environment variables
        self.n_results = int(os.getenv('N_RESULTS', 5))
        self.distance_threshold = float(os.getenv('DISTANCE_THRESHOLD', 1.5))
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L12-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n##", "\n\n", "\n", ". ", " ", ""]
        )
        logger.info(f"Initialized ChromaDocStore with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    @staticmethod
    def extract_text_from_pdf(pdf_file) -> List[Dict[str, any]]:
        """
        Extract text from a PDF file or file object
        Returns: List of dicts with 'text' and metadata for each page
        """
        try:
            pdf_reader = PdfReader(pdf_file)
            documents = []
            file_name = getattr(pdf_file, 'name', 'unknown')
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():  # Only include non-empty pages
                    documents.append({
                        'text': text,
                        'page_number': page_num,
                        'file_name': file_name
                    })
            return documents
        except Exception as e:
            logger.error(f"Error processing PDF {getattr(pdf_file, 'name', 'unknown')}: {str(e)}")
            raise

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
            
            # Ensure required metadata fields exist
            for metadata in metadatas:
                if 'file_name' not in metadata:
                    metadata['file_name'] = 'unknown'
                if 'page_range' not in metadata:
                    metadata['page_range'] = 'unknown'
            
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
    def query_documents(self, query: str, n_results: int = None, distance_threshold: float = None):
        """
        Query documents with a distance threshold to filter out irrelevant results.
        Lower distance means more similar (better match). Range is typically 0-1.
        
        Args:
            query (str): The query text to search for
            n_results (int, optional): Number of results to return. Defaults to self.n_results
            distance_threshold (float, optional): Maximum distance threshold for results. Defaults to self.distance_threshold
        """
        if n_results is None:
            n_results = self.n_results
        
        if distance_threshold is None:
            distance_threshold = self.distance_threshold
            
        logger.info(f"Querying documents with: {query[:100]}...")
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Filter out results above the distance threshold if distances are available
        if results['documents'] and results['documents'][0]:
            # Check if distances are available in results
            if 'distances' in results and results['distances'][0]:
                filtered_indices = [
                    i for i, dist in enumerate(results['distances'][0]) 
                    if dist <= distance_threshold
                ]
                
                # If no documents pass the threshold, return empty results
                if not filtered_indices:
                    logger.info("No documents found within acceptable distance threshold")
                    return {
                        'documents': [[]],
                        'metadatas': [[]],
                        'distances': [[]] if 'distances' in results else None
                    }
                
                # Filter all result lists to only include relevant documents
                results['documents'][0] = [results['documents'][0][i] for i in filtered_indices]
                results['metadatas'][0] = [results['metadatas'][0][i] for i in filtered_indices]
                if 'distances' in results:
                    results['distances'][0] = [results['distances'][0][i] for i in filtered_indices]
            
            # Log retrieved chunks and their distances
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i] if 'distances' in results else 'N/A'
                metadata = results['metadatas'][0][i]
                logger.info(f"Retrieved chunk {i + 1}/{len(results['documents'][0])}:")
                logger.info(f"  Distance: {distance}")
                logger.info(f"  Metadata: {metadata}")
                logger.info(f"  Content: {results['documents'][0][i][:50]}...")
        
        return results

    @log_time(logger)
    def clear_documents(self):
        logger.info("Clearing all documents and reinitializing collection")
        try:
            # Delete the entire collection
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            
            # Recreate the collection with the current embedding function
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Recreated collection: {self.collection_name}")
            
            return True
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return False

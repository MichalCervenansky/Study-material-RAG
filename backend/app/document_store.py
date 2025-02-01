import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from .logger_config import get_logger, log_time

logger = get_logger(__name__)

class ChromaDocStore:
    def __init__(self):
        self.settings = Settings(
            allow_reset=True,
            anonymized_telemetry=False,
            is_persistent=True
        )
        
        self.client = chromadb.Client(self.settings)
        self.collection_name = "documents"
        # Use default embedding function from ChromaDB
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.info("Initialized ChromaDocStore")

    @log_time(logger)
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> bool:
        logger.info(f"Adding {len(documents)} documents")
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def get_all_documents(self):
        return self.collection.get()

    @log_time(logger)
    def query_documents(self, query: str, n_results: int = 5):
        logger.info(f"Querying documents with: {query[:100]}...")
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

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

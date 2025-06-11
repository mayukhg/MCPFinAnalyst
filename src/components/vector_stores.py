"""
Wrapper for different Vector DBs (Chroma, Pinecone, etc.)
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ChromaDB imports
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

# OpenAI for embeddings
from openai import OpenAI
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

class VectorStoreManager:
    """Vector database interface using ChromaDB for document storage and retrieval."""
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self.collection = None
        self.openai_client = OpenAI(api_key=config.get_api_key("openai"))
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the ChromaDB client and collection."""
        if not chromadb:
            raise ImportError("ChromaDB is required. Install with: pip install chromadb")
        
        try:
            # Create data directory if it doesn't exist
            db_path = Path(self.config.vector_store.path)
            db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.vector_store.collection_name
                )
                logger.info(f"Connected to existing collection: {self.config.vector_store.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.config.vector_store.collection_name,
                    metadata={"description": "Agentic RAG document collection"}
                )
                logger.info(f"Created new collection: {self.config.vector_store.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def add_documents(self, chunks: List[Dict[str, Any]], source_file: str):
        """
        Add document chunks to the vector database.
        
        Args:
            chunks: List of document chunks with text and metadata
            source_file: Source file path for the chunks
        """
        if not chunks:
            logger.warning("No chunks to add to vector store")
            return
        
        try:
            logger.info(f"Adding {len(chunks)} chunks from {source_file} to vector store")
            
            # Prepare data for insertion
            ids = []
            texts = []
            metadatas = []
            embeddings = []
            
            for i, chunk in enumerate(chunks):
                # Generate unique ID for chunk
                chunk_id = f"{self._generate_file_id(source_file)}_{i}"
                ids.append(chunk_id)
                
                # Extract text
                text = chunk.get('text', '')
                texts.append(text)
                
                # Prepare metadata
                metadata = {
                    'source': source_file,
                    'chunk_index': i,
                    'char_count': len(text),
                    'word_count': len(text.split()),
                }
                
                # Add original metadata
                if chunk.get('metadata'):
                    metadata.update(chunk['metadata'])
                
                # Add page information if available
                if chunk.get('page'):
                    metadata['page'] = chunk['page']
                
                metadatas.append(metadata)
                
                # Generate embedding
                embedding = self._generate_embedding(text)
                embeddings.append(embedding)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in the vector database.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of similar document chunks with metadata
        """
        if k is None:
            k = self.config.retrieval.top_k
        
        try:
            logger.info(f"Performing similarity search for: {query}")
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'source': results['metadatas'][0][i].get('source', 'Unknown'),
                        'page': results['metadatas'][0][i].get('page'),
                        'chunk_index': results['metadatas'][0][i].get('chunk_index', 0)
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI's embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Clean text
            text = text.replace("\n", " ").strip()
            if not text:
                # Return zero vector for empty text
                return [0.0] * 1536  # text-embedding-3-large dimension
            
            response = self.openai_client.embeddings.create(
                model=self.config.embeddings.model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    def _generate_file_id(self, file_path: str) -> str:
        """Generate a consistent ID for a file."""
        import hashlib
        return hashlib.md5(file_path.encode()).hexdigest()[:8]
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
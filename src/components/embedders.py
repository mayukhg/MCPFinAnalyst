"""
Wrapper for different embedding models (OpenAI, HuggingFace, etc.)
"""

import os
from typing import List, Optional
from abc import ABC, abstractmethod
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseEmbedder(ABC):
    """Abstract base class for embedding implementations."""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        pass

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding implementation."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimension = 1536 if "large" in model else 512
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding using OpenAI's API."""
        try:
            # Clean text
            text = text.replace("\n", " ").strip()
            if not text:
                # Return zero vector for empty text
                return [0.0] * self._dimension
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            # Return zero vector as fallback
            return [0.0] * self._dimension
    
    def get_dimension(self) -> int:
        """Get the dimension of OpenAI embeddings."""
        return self._dimension

class EmbedderManager:
    """Manager class for different embedding implementations."""
    
    def __init__(self, config):
        self.config = config
        self.embedder = self._initialize_embedder()
    
    def _initialize_embedder(self) -> BaseEmbedder:
        """Initialize the appropriate embedder based on configuration."""
        model_name = self.config.embeddings.model.lower()
        
        if "openai" in model_name or "text-embedding" in model_name:
            api_key = self.config.get_api_key("openai")
            return OpenAIEmbedder(api_key=api_key, model=self.config.embeddings.model)
        else:
            # Default to OpenAI for now
            logger.warning(f"Unknown embedding model {self.config.embeddings.model}, defaulting to OpenAI")
            api_key = self.config.get_api_key("openai")
            return OpenAIEmbedder(api_key=api_key, model="text-embedding-3-large")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding using the configured embedder."""
        return self.embedder.generate_embedding(text)
    
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.embedder.get_dimension()
"""
Configuration management for the Agentic RAG Workflow Engine.
"""

import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class LLMConfig(BaseModel):
    """Configuration for Language Model."""
    model: str = Field(default="gpt-4o", description="LLM model name")
    temperature: float = Field(default=0.1, description="Temperature for generation")
    max_tokens: int = Field(default=2000, description="Maximum tokens in response")
    api_key_env: str = Field(default="OPENAI_API_KEY", description="Environment variable for API key")

class EmbeddingsConfig(BaseModel):
    """Configuration for embeddings model."""
    model: str = Field(default="text-embedding-3-large", description="Embeddings model name")
    api_key_env: str = Field(default="OPENAI_API_KEY", description="Environment variable for API key")

class VectorStoreConfig(BaseModel):
    """Configuration for vector database."""
    type: str = Field(default="chromadb", description="Vector store type")
    path: str = Field(default="./vector_db", description="Path to vector database")
    collection_name: str = Field(default="documents", description="Collection name")

class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    chunk_size: int = Field(default=1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    semantic_chunking: bool = Field(default=True, description="Use semantic chunking")

class RetrievalConfig(BaseModel):
    """Configuration for document retrieval."""
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")

class Config(BaseModel):
    """Main configuration class."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    
    @classmethod
    def load_config(cls, config_path: str = "config.yaml") -> "Config":
        """Load configuration from file or create default."""
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                return cls(**config_data)
        else:
            # Create default config
            config = cls()
            config.save_config(config_path)
            return config
    
    def save_config(self, config_path: str = "config.yaml"):
        """Save current configuration to file."""
        config_data = self.model_dump()
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def get_api_key(self, service: str) -> str:
        """Get API key for specified service."""
        if service == "openai":
            api_key = os.getenv(self.llm.api_key_env)
            if not api_key:
                raise ValueError(f"OpenAI API key not found in environment variable: {self.llm.api_key_env}")
            return api_key
        else:
            raise ValueError(f"Unknown service: {service}")

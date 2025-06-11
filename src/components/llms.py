"""
Wrapper for different LLMs (OpenAI, Llama, etc.)
"""

import os
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate a completion from the LLM."""
        pass

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate a completion using OpenAI's API."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if response_format:
                kwargs["response_format"] = response_format
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(**kwargs)
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise

class LLMManager:
    """Manager class for different LLM implementations."""
    
    def __init__(self, config):
        self.config = config
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> BaseLLM:
        """Initialize the appropriate LLM based on configuration."""
        model_name = self.config.llm.model.lower()
        
        if "gpt" in model_name or "openai" in model_name:
            api_key = self.config.get_api_key("openai")
            return OpenAILLM(api_key=api_key, model=self.config.llm.model)
        else:
            # Default to OpenAI for now
            logger.warning(f"Unknown model {self.config.llm.model}, defaulting to OpenAI")
            api_key = self.config.get_api_key("openai")
            return OpenAILLM(api_key=api_key, model="gpt-4o")
    
    def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate a completion using the configured LLM."""
        temp = temperature if temperature is not None else self.config.llm.temperature
        tokens = max_tokens if max_tokens is not None else self.config.llm.max_tokens
        
        return self.llm.generate_completion(
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
            response_format=response_format
        )
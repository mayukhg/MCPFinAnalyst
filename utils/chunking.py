"""
Document chunking utilities for semantic and simple text chunking strategies.
"""

import re
import math
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseChunker(ABC):
    """Abstract base class for document chunking strategies."""
    
    def __init__(self, config: Config):
        self.config = config
        self.chunk_size = config.chunking.chunk_size
        self.chunk_overlap = config.chunking.chunk_overlap
    
    @abstractmethod
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            text: Document text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunks with metadata
        """
        pass
    
    def _extract_page_info(self, text: str) -> List[Dict[str, Any]]:
        """Extract page information from text with page markers."""
        pages = []
        page_pattern = r'\[PAGE (\d+)\]'
        
        # Split text by page markers
        parts = re.split(page_pattern, text)
        
        if len(parts) == 1:
            # No page markers found
            pages.append({
                'page_number': None,
                'content': text.strip()
            })
        else:
            # Process parts with page numbers
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    page_number = int(parts[i])
                    content = parts[i + 1].strip()
                    if content:
                        pages.append({
                            'page_number': page_number,
                            'content': content
                        })
        
        return pages
    
    def _create_chunk(
        self, 
        text: str, 
        chunk_index: int, 
        metadata: Dict[str, Any], 
        page_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        chunk = {
            'text': text.strip(),
            'chunk_index': chunk_index,
            'metadata': metadata.copy()
        }
        
        if page_number:
            chunk['page'] = page_number
        
        # Add chunk-specific metadata
        chunk['metadata'].update({
            'chunk_char_count': len(text),
            'chunk_word_count': len(text.split()),
            'chunk_index': chunk_index
        })
        
        return chunk

class SimpleChunker(BaseChunker):
    """Simple text chunking based on fixed character counts with overlap."""
    
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document using simple character-based splitting.
        
        Args:
            text: Document text to chunk
            metadata: Document metadata
            
        Returns:
            List of text chunks with metadata
        """
        try:
            logger.info(f"Chunking document with simple strategy: {len(text)} characters")
            
            # Extract page information
            pages = self._extract_page_info(text)
            
            chunks = []
            chunk_index = 0
            
            for page_info in pages:
                page_text = page_info['content']
                page_number = page_info['page_number']
                
                # Split page text into chunks
                page_chunks = self._split_text_simple(page_text)
                
                for chunk_text in page_chunks:
                    chunk = self._create_chunk(
                        chunk_text, 
                        chunk_index, 
                        metadata, 
                        page_number
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            
            logger.info(f"Created {len(chunks)} chunks using simple strategy")
            return chunks
            
        except Exception as e:
            logger.error(f"Simple chunking failed: {e}")
            raise
    
    def _split_text_simple(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
        
        return chunks

class SemanticChunker(BaseChunker):
    """Semantic chunking that tries to keep related content together."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        # Patterns for semantic boundaries
        self.paragraph_pattern = r'\n\s*\n'
        self.sentence_pattern = r'[.!?]+\s+'
        self.section_patterns = [
            r'\n#{1,6}\s+',  # Markdown headers
            r'\n\d+\.\s+',   # Numbered lists
            r'\n[-*+]\s+',   # Bullet points
            r'\n[A-Z][A-Z\s]{10,}:\s*\n',  # All caps headers
        ]
    
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document using semantic boundaries.
        
        Args:
            text: Document text to chunk
            metadata: Document metadata
            
        Returns:
            List of semantically coherent chunks
        """
        try:
            logger.info(f"Chunking document with semantic strategy: {len(text)} characters")
            
            # Extract page information
            pages = self._extract_page_info(text)
            
            chunks = []
            chunk_index = 0
            
            for page_info in pages:
                page_text = page_info['content']
                page_number = page_info['page_number']
                
                # Split page text into semantic chunks
                page_chunks = self._split_text_semantic(page_text)
                
                for chunk_text in page_chunks:
                    chunk = self._create_chunk(
                        chunk_text, 
                        chunk_index, 
                        metadata, 
                        page_number
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            
            logger.info(f"Created {len(chunks)} chunks using semantic strategy")
            return chunks
            
        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}")
            # Fallback to simple chunking
            logger.info("Falling back to simple chunking strategy")
            simple_chunker = SimpleChunker(self.config)
            return simple_chunker.chunk_document(text, metadata)
    
    def _split_text_semantic(self, text: str) -> List[str]:
        """Split text using semantic boundaries."""
        # First, try to split by sections
        sections = self._split_by_sections(text)
        
        chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # Section is too large, split by paragraphs
                section_chunks = self._split_large_section(section)
                chunks.extend(section_chunks)
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by semantic section boundaries."""
        # Try each section pattern
        for pattern in self.section_patterns:
            if re.search(pattern, text):
                sections = re.split(pattern, text)
                # Filter out empty sections and rejoin split markers
                filtered_sections = []
                for i, section in enumerate(sections):
                    section = section.strip()
                    if section:
                        # Add back the section marker (except for the first section)
                        if i > 0:
                            match = re.search(pattern, text)
                            if match:
                                section = match.group() + section
                        filtered_sections.append(section)
                return filtered_sections
        
        # No section patterns found, split by paragraphs
        return self._split_by_paragraphs(text)
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraph boundaries."""
        paragraphs = re.split(self.paragraph_pattern, text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_large_section(self, section: str) -> List[str]:
        """Split a large section into smaller chunks."""
        # First try paragraphs
        paragraphs = self._split_by_paragraphs(section)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Handle large paragraphs
                if len(paragraph) > self.chunk_size:
                    # Split large paragraph by sentences
                    paragraph_chunks = self._split_by_sentences(paragraph)
                    chunks.extend(paragraph_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentence boundaries."""
        sentences = re.split(self.sentence_pattern, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add sentence ending punctuation back
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Handle very long sentences
                if len(sentence) > self.chunk_size:
                    # Split by words as last resort
                    word_chunks = self._split_by_words(sentence)
                    chunks.extend(word_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_words(self, text: str) -> List[str]:
        """Split text by words (last resort for very long sentences)."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length <= self.chunk_size:
                current_chunk.append(word)
                current_length += word_length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure for better chunking decisions."""
        structure = {
            'total_chars': len(text),
            'total_words': len(text.split()),
            'paragraph_count': len(re.split(self.paragraph_pattern, text)),
            'sentence_count': len(re.split(self.sentence_pattern, text)),
            'has_sections': False,
            'section_types': []
        }
        
        # Check for section patterns
        for i, pattern in enumerate(self.section_patterns):
            if re.search(pattern, text):
                structure['has_sections'] = True
                structure['section_types'].append(f"pattern_{i}")
        
        # Estimate optimal chunk count
        target_chunk_size = self.chunk_size * 0.8  # Aim for 80% of max size
        estimated_chunks = math.ceil(len(text) / target_chunk_size)
        structure['estimated_chunks'] = estimated_chunks
        
        return structure

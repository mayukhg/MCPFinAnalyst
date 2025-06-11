"""
Loads documents from various sources and formats.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Document processing libraries
try:
    import PyPDF2
    import pdfplumber
except ImportError:
    PyPDF2 = None
    pdfplumber = None

import markdown
from markdown.extensions import codehilite, fenced_code

from processing.chunker import SemanticChunker, SimpleChunker
from utils.logger import get_logger

logger = get_logger(__name__)

class DocumentLoader:
    """Handles document loading, processing, and chunking."""
    
    def __init__(self, config):
        self.config = config
        self.chunker = self._initialize_chunker()
    
    def _initialize_chunker(self):
        """Initialize the appropriate chunking strategy."""
        if self.config.chunking.semantic_chunking:
            return SemanticChunker(self.config)
        else:
            return SimpleChunker(self.config)
    
    def find_documents(self, documents_path: Path, recursive: bool = False) -> List[Path]:
        """
        Find all supported documents in a directory.
        
        Args:
            documents_path: Path to search for documents
            recursive: Whether to search recursively
            
        Returns:
            List of document file paths
        """
        supported_extensions = {'.pdf', '.md', '.txt'}
        documents = []
        
        if documents_path.is_file():
            if documents_path.suffix.lower() in supported_extensions:
                documents = [documents_path]
        else:
            pattern = "**/*" if recursive else "*"
            for ext in supported_extensions:
                documents.extend(documents_path.glob(f"{pattern}{ext}"))
        
        return documents
    
    def load_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load and process a document into chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks with metadata
        """
        try:
            logger.info(f"Loading document: {file_path}")
            
            # Check if file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Extract text based on file type
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.pdf':
                text_content = self._extract_pdf_text(file_path)
            elif file_extension == '.md':
                text_content = self._extract_markdown_text(file_path)
            elif file_extension == '.txt':
                text_content = self._extract_text_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            if not text_content.strip():
                logger.warning(f"No text content extracted from {file_path}")
                return []
            
            # Generate document metadata
            doc_metadata = self._generate_document_metadata(file_path, text_content)
            
            # Chunk the document
            chunks = self.chunker.chunk_document(text_content, doc_metadata)
            
            logger.info(f"Successfully loaded {file_path}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        if not pdfplumber:
            raise ImportError("PDF processing requires 'pdfplumber'. Install with: pip install pdfplumber")
        
        text_content = []
        page_number = 1
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # Add page marker for citation purposes
                        text_content.append(f"[PAGE {page_number}]\n{page_text.strip()}")
                    page_number += 1
        except Exception as e:
            logger.warning(f"pdfplumber failed for {file_path}, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            if PyPDF2:
                try:
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        for page_num, page in enumerate(pdf_reader.pages, 1):
                            page_text = page.extract_text()
                            if page_text:
                                text_content.append(f"[PAGE {page_num}]\n{page_text.strip()}")
                except Exception as e2:
                    raise Exception(f"Both PDF extraction methods failed: pdfplumber: {e}, PyPDF2: {e2}")
            else:
                raise Exception(f"PDF extraction failed and PyPDF2 not available: {e}")
        
        return "\n\n".join(text_content)
    
    def _extract_markdown_text(self, file_path: Path) -> str:
        """Extract text from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                markdown_content = file.read()
            
            # Convert markdown to plain text (removing formatting)
            md = markdown.Markdown(extensions=['codehilite', 'fenced_code'])
            html_content = md.convert(markdown_content)
            
            # Simple HTML tag removal for plain text
            import re
            text_content = re.sub(r'<[^>]+>', '', html_content)
            text_content = re.sub(r'\n\s*\n', '\n\n', text_content)  # Normalize whitespace
            
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"Error extracting markdown from {file_path}: {e}")
            # Fallback: return raw markdown
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
    
    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from plain text file."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            
            raise UnicodeDecodeError(f"Could not decode {file_path} with any supported encoding")
            
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            raise
    
    def _generate_document_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Generate metadata for the document."""
        # Calculate content hash for deduplication
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # Get file stats
        file_stats = file_path.stat()
        
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': file_path.suffix.lower(),
            'file_size': file_stats.st_size,
            'content_hash': content_hash,
            'char_count': len(content),
            'word_count': len(content.split()),
            'created_timestamp': file_stats.st_ctime,
            'modified_timestamp': file_stats.st_mtime
        }
        
        return metadata
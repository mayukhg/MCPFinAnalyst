# API Documentation

Programming interface for the Agentic RAG Workflow Engine components.

## Core API Classes

### WorkflowOrchestrator

Main orchestrator for executing the multi-agent RAG workflow.

```python
from core.orchestrator import WorkflowOrchestrator
from utils.config_manager import ConfigManager

# Initialize
config_manager = ConfigManager('config.yaml')
orchestrator = WorkflowOrchestrator(config_manager.config)

# Execute workflow
result = orchestrator.execute_workflow(
    query="What are the main features?",
    verbose=True
)
```

#### Methods

**execute_workflow(query: str, verbose: bool = False) -> Dict[str, Any]**

Execute the complete agentic RAG workflow.

- **Parameters:**
  - `query`: User's question as string
  - `verbose`: Include detailed workflow steps in output
- **Returns:** Dictionary with answer, sources, and metadata
- **Example:**
```python
result = orchestrator.execute_workflow("How does authentication work?")
print(result['answer'])
for source in result['sources']:
    print(f"Source: {source['file']}")
```

**validate_workflow_config() -> Dict[str, Any]**

Validate workflow configuration and dependencies.

- **Returns:** Validation results with errors and warnings
- **Example:**
```python
validation = orchestrator.validate_workflow_config()
if not validation['valid']:
    print("Errors:", validation['errors'])
```

### DocumentLoader

Handles document ingestion and processing.

```python
from processing.loader import DocumentLoader
from utils.config_manager import ConfigManager

config_manager = ConfigManager('config.yaml')
loader = DocumentLoader(config_manager.config)
```

#### Methods

**load_document(file_path: Path) -> List[Dict[str, Any]]**

Load and process a single document.

- **Parameters:** `file_path`: Path to document file
- **Returns:** List of document chunks with metadata
- **Example:**
```python
from pathlib import Path
chunks = loader.load_document(Path('data/document.pdf'))
print(f"Created {len(chunks)} chunks")
```

**find_documents(documents_path: Path, recursive: bool = False) -> List[Path]**

Find supported documents in directory.

- **Parameters:**
  - `documents_path`: Directory to search
  - `recursive`: Search subdirectories
- **Returns:** List of document file paths
- **Example:**
```python
docs = loader.find_documents(Path('data/'), recursive=True)
for doc in docs:
    print(f"Found: {doc.name}")
```

### VectorStoreManager

Manages vector database operations.

```python
from components.vector_stores import VectorStoreManager
from utils.config_manager import ConfigManager

config_manager = ConfigManager('config.yaml')
vector_store = VectorStoreManager(config_manager.config)
```

#### Methods

**add_documents(chunks: List[Dict[str, Any]], source_file: str)**

Add document chunks to vector database.

- **Parameters:**
  - `chunks`: List of document chunks with text and metadata
  - `source_file`: Source file path for chunks
- **Example:**
```python
vector_store.add_documents(chunks, 'data/document.pdf')
```

**similarity_search(query: str, k: int = None, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]**

Perform similarity search in vector database.

- **Parameters:**
  - `query`: Search query
  - `k`: Number of results to return
  - `filter_dict`: Optional metadata filters
- **Returns:** List of similar document chunks
- **Example:**
```python
results = vector_store.similarity_search(
    "authentication methods", 
    k=5
)
for result in results:
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Text: {result['text'][:100]}...")
```

**get_document_count() -> int**

Get total number of documents in collection.

- **Returns:** Document count
- **Example:**
```python
count = vector_store.get_document_count()
print(f"Total documents: {count}")
```

### Agent Classes

#### QueryAnalyzer

Analyzes user queries for optimal retrieval.

```python
from core.agents import QueryAnalyzer
from utils.config_manager import ConfigManager

config_manager = ConfigManager('config.yaml')
analyzer = QueryAnalyzer(config_manager.config)
```

**analyze_query(query: str) -> Dict[str, Any]**

Analyze query structure and extract key information.

- **Parameters:** `query`: User's question
- **Returns:** Structured analysis object
- **Example:**
```python
analysis = analyzer.analyze_query("Compare neural network architectures")
print(f"Complexity: {analysis['complexity_score']}/5")
print(f"Entities: {analysis['key_entities']}")
```

#### Retriever

Performs intelligent document retrieval.

```python
from core.agents import Retriever
retriever = Retriever(config_manager.config)
```

**retrieve_documents(query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]**

Retrieve relevant documents based on query analysis.

- **Parameters:**
  - `query`: Original user query
  - `analysis`: Query analysis from QueryAnalyzer
- **Returns:** List of relevant document chunks
- **Example:**
```python
docs = retriever.retrieve_documents(query, analysis)
context = retriever.get_retrieval_context(docs)
```

#### Synthesizer

Generates answers from retrieved context.

```python
from core.agents import Synthesizer
synthesizer = Synthesizer(config_manager.config)
```

**synthesize_answer(query: str, context: str, analysis: Dict, retrieved_docs: List) -> Dict[str, Any]**

Generate comprehensive answer with citations.

- **Parameters:**
  - `query`: Original user query
  - `context`: Retrieved document context
  - `analysis`: Query analysis
  - `retrieved_docs`: Retrieved documents for citation
- **Returns:** Synthesized answer with metadata
- **Example:**
```python
result = synthesizer.synthesize_answer(
    query=query,
    context=context,
    analysis=analysis,
    retrieved_docs=docs
)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Configuration API

### ConfigManager

Manages system configuration.

```python
from utils.config_manager import ConfigManager

# Load configuration
config_manager = ConfigManager('config.yaml')
config = config_manager.config

# Access configuration values
print(f"LLM Model: {config.llm.model}")
print(f"Chunk Size: {config.chunking.chunk_size}")
```

#### Methods

**load_config() -> Config**

Load configuration from file.

**save_config(config: Config)**

Save configuration to file.

**update_config(updates: Dict[str, Any])**

Update configuration with new values.

- **Example:**
```python
updates = {
    'retrieval': {
        'top_k': 10,
        'similarity_threshold': 0.6
    }
}
config_manager.update_config(updates)
```

## Data Structures

### Query Analysis Result

```python
{
    "query_type": str,           # "simple" | "complex" | "multi_part"
    "main_topic": str,           # Primary subject of query
    "key_entities": List[str],   # Important entities
    "sub_questions": List[str],  # Decomposed questions
    "search_terms": List[str],   # Optimized search terms
    "intent": str,               # User intent classification
    "complexity_score": int,     # 1-5 complexity rating
    "retrieval_strategy": str    # Recommended approach
}
```

### Retrieved Document

```python
{
    "id": str,                   # Unique chunk identifier
    "text": str,                 # Document chunk text
    "source": str,               # Source file path
    "page": Optional[int],       # Page number if applicable
    "similarity_score": float,   # Vector similarity score
    "boosted_score": float,      # After intelligent ranking
    "metadata": Dict[str, Any]   # Additional metadata
}
```

### Workflow Result

```python
{
    "answer": str,               # Generated response
    "sources": List[Dict],       # Source citations
    "confidence": float,         # Answer confidence (0-1)
    "query_analysis": Dict,      # Original query analysis
    "processing_time": float,    # Total processing time
    "workflow_steps": List[Dict] # Detailed steps (if verbose)
}
```

## Advanced Usage Examples

### Custom Workflow Implementation

```python
from core.orchestrator import WorkflowOrchestrator
from utils.config_manager import ConfigManager

class CustomRAGWorkflow:
    def __init__(self, config_path='config.yaml'):
        self.config_manager = ConfigManager(config_path)
        self.orchestrator = WorkflowOrchestrator(self.config_manager.config)
    
    def process_document_batch(self, file_paths):
        """Process multiple documents efficiently."""
        from processing.loader import DocumentLoader
        from components.vector_stores import VectorStoreManager
        
        loader = DocumentLoader(self.config_manager.config)
        vector_store = VectorStoreManager(self.config_manager.config)
        
        for file_path in file_paths:
            try:
                chunks = loader.load_document(file_path)
                vector_store.add_documents(chunks, str(file_path))
                print(f"Processed: {file_path.name}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    def query_with_filters(self, query, source_filter=None):
        """Query with metadata filtering."""
        # Custom implementation using individual agents
        pass
    
    def get_workflow_analytics(self):
        """Get workflow performance analytics."""
        return self.orchestrator.get_workflow_statistics()

# Usage
workflow = CustomRAGWorkflow()
workflow.process_document_batch([Path('doc1.pdf'), Path('doc2.md')])
result = workflow.orchestrator.execute_workflow("What are the main points?")
```

### Batch Query Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BatchQueryProcessor:
    def __init__(self, config_path='config.yaml'):
        self.config_manager = ConfigManager(config_path)
        self.orchestrator = WorkflowOrchestrator(self.config_manager.config)
    
    def process_queries(self, queries, max_workers=3):
        """Process multiple queries in parallel."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.orchestrator.execute_workflow, query)
                for query in queries
            ]
            
            results = []
            for future, query in zip(futures, queries):
                try:
                    result = future.result()
                    results.append({
                        'query': query,
                        'answer': result['answer'],
                        'confidence': result['confidence']
                    })
                except Exception as e:
                    results.append({
                        'query': query,
                        'error': str(e)
                    })
            
            return results

# Usage
processor = BatchQueryProcessor()
queries = [
    "What are the main features?",
    "How does authentication work?",
    "What are the performance metrics?"
]
results = processor.process_queries(queries)
```

### Custom Component Integration

```python
from components.llms import BaseLLM, LLMManager
from components.embedders import BaseEmbedder, EmbedderManager

class CustomLLM(BaseLLM):
    """Custom LLM implementation."""
    
    def generate_completion(self, messages, temperature=0.1, max_tokens=2000, response_format=None):
        # Custom LLM implementation
        # Could integrate with local models, other APIs, etc.
        pass

class CustomEmbedder(BaseEmbedder):
    """Custom embedding implementation."""
    
    def generate_embedding(self, text):
        # Custom embedding implementation
        pass
    
    def get_dimension(self):
        return 768  # Custom dimension

# Integration
config_manager = ConfigManager('config.yaml')

# Override with custom components
llm_manager = LLMManager(config_manager.config)
llm_manager.llm = CustomLLM()

embedder_manager = EmbedderManager(config_manager.config)
embedder_manager.embedder = CustomEmbedder()
```

## Error Handling

### Exception Types

The API raises specific exceptions for different error conditions:

```python
from utils.exceptions import (
    ConfigurationError,
    DocumentProcessingError,
    VectorStoreError,
    QueryProcessingError
)

try:
    result = orchestrator.execute_workflow("complex query")
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
except DocumentProcessingError as e:
    print(f"Document processing failed: {e}")
except VectorStoreError as e:
    print(f"Vector store error: {e}")
except QueryProcessingError as e:
    print(f"Query processing failed: {e}")
```

### Graceful Error Handling

```python
def safe_query(orchestrator, query, max_retries=3):
    """Execute query with retry logic."""
    for attempt in range(max_retries):
        try:
            return orchestrator.execute_workflow(query)
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    'answer': f"Failed to process query after {max_retries} attempts: {e}",
                    'sources': [],
                    'confidence': 0.0,
                    'error': str(e)
                }
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Performance Monitoring

### API Performance Tracking

```python
import time
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor API call performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            print(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            print(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper

# Usage
@monitor_performance
def timed_query(orchestrator, query):
    return orchestrator.execute_workflow(query)
```
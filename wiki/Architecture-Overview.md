# Architecture Overview

The Agentic RAG Workflow Engine implements a sophisticated multi-agent architecture that goes beyond traditional RAG systems. This document explains the core components and their interactions.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│               Workflow Orchestrator                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │   Query     │ │  Retriever  │ │     Synthesizer         ││
│  │  Analyzer   │ │   Agent     │ │       Agent             ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                Component Layer                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │  LLM        │ │   Vector    │ │      Embedder           ││
│  │ Manager     │ │   Store     │ │      Manager            ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│              Processing Layer                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │  Document   │ │  Semantic   │ │    Configuration        ││
│  │   Loader    │ │  Chunker    │ │      Manager            ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Workflow Orchestrator

The central coordinator that manages the multi-agent workflow execution.

**Responsibilities:**
- Coordinates agent interactions
- Manages workflow state and timing
- Handles error recovery and fallbacks
- Provides workflow observability

**Key Methods:**
- `execute_workflow()` - Main workflow execution
- `validate_workflow_config()` - Configuration validation
- `get_workflow_statistics()` - Performance metrics

### 2. Agent Layer

#### Query Analyzer Agent
Breaks down user queries into structured analysis for optimal retrieval.

**Core Functions:**
- Entity extraction from queries
- Query complexity assessment
- Sub-question generation for complex queries
- Search term optimization

**Analysis Output:**
```json
{
  "query_type": "simple|complex|multi_part",
  "main_topic": "primary subject",
  "key_entities": ["entity1", "entity2"],
  "sub_questions": ["question1", "question2"],
  "search_terms": ["term1", "term2"],
  "intent": "information_seeking",
  "complexity_score": 1-5,
  "retrieval_strategy": "recommended approach"
}
```

#### Retriever Agent
Performs intelligent document retrieval based on query analysis.

**Features:**
- Multi-query search strategies
- Intelligent result ranking
- Entity-based score boosting
- Similarity threshold filtering

**Retrieval Process:**
1. Generate multiple search queries from analysis
2. Perform vector similarity search
3. Apply intelligent ranking algorithms
4. Filter and deduplicate results

#### Synthesizer Agent
Generates comprehensive answers with proper source citations.

**Capabilities:**
- Context-aware answer generation
- Source attribution and citation
- Confidence estimation
- Answer quality validation

## Component Layer

### LLM Manager
Provides abstraction for different language models.

**Supported Models:**
- OpenAI GPT-4o (default)
- Extensible for other providers

**Features:**
- Model switching via configuration
- Temperature and token control
- Response format handling

### Vector Store Manager
Handles document storage and similarity search.

**Current Implementation:**
- ChromaDB for persistent storage
- OpenAI embeddings integration
- Metadata filtering capabilities

**Storage Features:**
- Persistent vector database
- Automatic embedding generation
- Document metadata tracking

### Embedder Manager
Manages text embedding generation.

**Capabilities:**
- OpenAI embedding models
- Configurable embedding dimensions
- Fallback handling for failures

## Processing Layer

### Document Loader
Handles multi-format document ingestion.

**Supported Formats:**
- PDF files (via pdfplumber/PyPDF2)
- Markdown files (with HTML conversion)
- Plain text files (multiple encodings)

**Features:**
- Recursive directory processing
- Metadata extraction
- Content validation

### Semantic Chunker
Intelligent document chunking that preserves context.

**Chunking Strategies:**
- **Semantic Chunking**: Preserves logical boundaries
- **Simple Chunking**: Fixed-size with overlap

**Semantic Boundaries:**
- Paragraph breaks
- Section headers
- List items
- Sentence boundaries

## Data Flow

### Document Ingestion Flow
```
Documents → Loader → Chunker → Embedder → Vector Store
```

1. **Document Discovery**: Find supported files
2. **Text Extraction**: Extract content by format
3. **Semantic Chunking**: Split preserving context
4. **Embedding Generation**: Create vector representations
5. **Storage**: Persist in vector database

### Query Processing Flow
```
Query → Analyzer → Retriever → Synthesizer → Response
```

1. **Query Analysis**: Extract entities and intent
2. **Strategic Retrieval**: Multi-query search approach
3. **Context Assembly**: Format retrieved chunks
4. **Answer Synthesis**: Generate cited response
5. **Quality Assessment**: Confidence scoring

## Configuration System

### YAML Configuration
```yaml
llm:
  model: gpt-4o
  temperature: 0.1
  max_tokens: 2000

embeddings:
  model: text-embedding-3-large

vector_store:
  type: chromadb
  path: ./vector_db
  collection_name: documents

chunking:
  chunk_size: 1000
  chunk_overlap: 200
  semantic_chunking: true

retrieval:
  top_k: 5
  similarity_threshold: 0.7
```

### Configuration Management
- Environment variable integration
- Runtime configuration updates
- Validation and error handling

## Performance Characteristics

### Scalability Metrics
- **Document Capacity**: 1000+ documents
- **Query Response Time**: 5-10 seconds
- **Memory Usage**: Scales with document corpus
- **Concurrent Queries**: Single-threaded design

### Optimization Features
- Persistent vector storage
- Efficient similarity search
- Intelligent result caching
- Configurable retrieval limits

## Error Handling

### Graceful Degradation
- Fallback to simple chunking if semantic fails
- Alternative PDF extraction methods
- Default responses for API failures

### Observability
- Comprehensive logging
- Workflow step tracking
- Performance monitoring
- Error reporting

## Extension Points

### Adding New Components

**Custom LLM Providers:**
```python
class CustomLLM(BaseLLM):
    def generate_completion(self, messages, **kwargs):
        # Custom implementation
        pass
```

**Custom Vector Stores:**
```python
class CustomVectorStore(BaseVectorStore):
    def similarity_search(self, query, k):
        # Custom implementation
        pass
```

**Custom Embedding Models:**
```python
class CustomEmbedder(BaseEmbedder):
    def generate_embedding(self, text):
        # Custom implementation
        pass
```

## Security Considerations

### API Key Management
- Environment variable storage
- No hardcoded credentials
- Secure configuration loading

### Data Privacy
- Local vector storage option
- No data sent to third parties (except OpenAI API)
- Configurable data retention

## Future Architecture Enhancements

### Planned Improvements
- Distributed agent execution
- Multiple vector store backends
- Advanced caching strategies
- Real-time document updates
- Multi-modal document support
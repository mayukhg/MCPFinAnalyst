# Configuration

The Agentic RAG Workflow Engine uses YAML-based configuration for flexible system customization. This document covers all configuration options and best practices.

## Configuration File Structure

The main configuration file (`config.yaml`) is organized into logical sections:

```yaml
llm:
  model: gpt-4o
  temperature: 0.1
  max_tokens: 2000
  api_key_env: OPENAI_API_KEY

embeddings:
  model: text-embedding-3-large
  api_key_env: OPENAI_API_KEY

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

## Configuration Sections

### LLM Configuration

Controls the language model used for query analysis and answer synthesis.

```yaml
llm:
  model: gpt-4o                    # Model name
  temperature: 0.1                 # Creativity level (0.0-2.0)
  max_tokens: 2000                 # Maximum response length
  api_key_env: OPENAI_API_KEY      # Environment variable for API key
```

**Model Options:**
- `gpt-4o` - Latest OpenAI model (recommended)
- `gpt-4` - Previous generation GPT-4
- `gpt-3.5-turbo` - Faster, less capable option

**Temperature Guidelines:**
- `0.0-0.2` - Focused, deterministic responses
- `0.3-0.7` - Balanced creativity and consistency
- `0.8-2.0` - High creativity, less predictable

### Embeddings Configuration

Configures text embedding generation for vector search.

```yaml
embeddings:
  model: text-embedding-3-large    # Embedding model
  api_key_env: OPENAI_API_KEY      # API key environment variable
```

**Model Options:**
- `text-embedding-3-large` - High-dimensional, best quality (1536 dims)
- `text-embedding-3-small` - Smaller, faster option (1536 dims)
- `text-embedding-ada-002` - Legacy model (1536 dims)

### Vector Store Configuration

Manages document storage and similarity search.

```yaml
vector_store:
  type: chromadb                   # Vector database type
  path: ./vector_db                # Storage directory
  collection_name: documents       # Collection identifier
```

**Storage Options:**
- `chromadb` - Local persistent storage (current implementation)
- Future: `pinecone`, `weaviate`, `qdrant`

**Path Configuration:**
- Relative paths are relative to project root
- Absolute paths are supported
- Directory will be created automatically

### Chunking Configuration

Controls document processing and chunking behavior.

```yaml
chunking:
  chunk_size: 1000                 # Target chunk size in characters
  chunk_overlap: 200               # Character overlap between chunks
  semantic_chunking: true          # Use intelligent chunking
```

**Chunk Size Guidelines:**
- `500-800` - Fine-grained, more precise retrieval
- `1000-1500` - Balanced context and precision
- `1500-2000` - Larger context, fewer chunks

**Overlap Recommendations:**
- `10-20%` of chunk size for good continuity
- Higher overlap for complex documents
- Lower overlap for simple, structured content

**Semantic Chunking:**
- `true` - Preserves logical boundaries (recommended)
- `false` - Simple character-based splitting

### Retrieval Configuration

Optimizes document retrieval behavior.

```yaml
retrieval:
  top_k: 5                         # Number of documents to retrieve
  similarity_threshold: 0.7        # Minimum similarity score
```

**Top-K Guidelines:**
- `3-5` - Focused retrieval for simple queries
- `5-10` - Comprehensive retrieval for complex queries
- `10+` - Extensive search for research applications

**Similarity Threshold:**
- `0.8-1.0` - Very strict matching
- `0.6-0.8` - Balanced relevance filtering
- `0.4-0.6` - Permissive matching

## Environment Variables

### Required Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Optional Variables

```bash
export RAG_CONFIG_PATH="/path/to/custom-config.yaml"
export RAG_LOG_LEVEL="INFO"
export RAG_VECTOR_DB_PATH="/custom/vector/path"
```

## Configuration Profiles

### Research Configuration

Optimized for academic and research use:

```yaml
# research-config.yaml
llm:
  model: gpt-4o
  temperature: 0.05
  max_tokens: 3000

retrieval:
  top_k: 10
  similarity_threshold: 0.6

chunking:
  chunk_size: 1500
  chunk_overlap: 300
  semantic_chunking: true
```

### Production Configuration

Optimized for speed and efficiency:

```yaml
# production-config.yaml
llm:
  model: gpt-4o
  temperature: 0.1
  max_tokens: 1500

retrieval:
  top_k: 5
  similarity_threshold: 0.75

chunking:
  chunk_size: 800
  chunk_overlap: 150
  semantic_chunking: true
```

### Development Configuration

For testing and development:

```yaml
# dev-config.yaml
llm:
  model: gpt-3.5-turbo
  temperature: 0.2
  max_tokens: 1000

retrieval:
  top_k: 3
  similarity_threshold: 0.6

chunking:
  chunk_size: 500
  chunk_overlap: 100
  semantic_chunking: false
```

## Dynamic Configuration

### Runtime Configuration Updates

```python
from utils.config_manager import ConfigManager

# Load configuration
config_manager = ConfigManager('config.yaml')

# Update settings
updates = {
    'retrieval': {
        'top_k': 10,
        'similarity_threshold': 0.6
    }
}
config_manager.update_config(updates)
```

### Custom Configuration Loading

```bash
# Use custom configuration
python src/cli.py --config research-config.yaml query "research question"

# Environment variable override
export RAG_CONFIG_PATH="production-config.yaml"
python src/cli.py status
```

## Validation and Defaults

### Configuration Validation

The system validates configuration on startup:

```python
# Automatic validation
config_manager = ConfigManager('config.yaml')
validation = config_manager.validate_config()

if not validation['valid']:
    print("Configuration errors:", validation['errors'])
```

### Default Values

If configuration file is missing, defaults are:

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

## Performance Tuning

### Memory Optimization

```yaml
# Reduce memory usage
chunking:
  chunk_size: 800
  chunk_overlap: 100

retrieval:
  top_k: 3
```

### Speed Optimization

```yaml
# Faster processing
llm:
  model: gpt-3.5-turbo
  max_tokens: 1000

chunking:
  semantic_chunking: false

retrieval:
  similarity_threshold: 0.8
```

### Quality Optimization

```yaml
# Higher quality results
llm:
  model: gpt-4o
  temperature: 0.05
  max_tokens: 3000

retrieval:
  top_k: 10
  similarity_threshold: 0.6

chunking:
  chunk_size: 1500
  semantic_chunking: true
```

## Security Configuration

### API Key Management

```yaml
# Use environment variables (recommended)
llm:
  api_key_env: OPENAI_API_KEY

# Never hardcode keys in configuration files
# ❌ Bad:
# llm:
#   api_key: sk-1234567890abcdef...
```

### Data Privacy

```yaml
# Local vector storage (no external services)
vector_store:
  type: chromadb
  path: ./private_vector_db

# Minimal external API usage
llm:
  model: gpt-4o  # Only for synthesis, not document storage
```

## Troubleshooting Configuration

### Common Issues

**Configuration not found:**
```bash
# Create default configuration
python src/cli.py status  # Auto-creates config.yaml
```

**Invalid model name:**
```bash
# Check available models in OpenAI documentation
# Update config.yaml with valid model name
```

**API key errors:**
```bash
# Verify environment variable
echo $OPENAI_API_KEY

# Check API key validity
python -c "import openai; openai.api_key='$OPENAI_API_KEY'"
```

### Configuration Testing

```bash
# Test configuration validity
python src/cli.py status

# Test with verbose output
python src/cli.py query "test question" --verbose

# Validate specific configuration
python src/cli.py --config test-config.yaml status
```

## Best Practices

### Configuration Management

1. **Version Control**: Include configuration files in version control
2. **Environment Separation**: Use different configs for dev/staging/production
3. **Documentation**: Comment complex configuration choices
4. **Validation**: Test configuration changes before deployment

### Security Best Practices

1. **Environment Variables**: Always use environment variables for secrets
2. **File Permissions**: Restrict access to configuration files
3. **Key Rotation**: Regularly rotate API keys
4. **Monitoring**: Log configuration changes and access

### Performance Best Practices

1. **Baseline Testing**: Measure performance with default settings
2. **Incremental Tuning**: Change one parameter at a time
3. **Load Testing**: Test with realistic document volumes
4. **Monitoring**: Track key performance metrics